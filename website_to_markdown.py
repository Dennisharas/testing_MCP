#!/usr/bin/env python3
"""Crawl a website and save its pages as a Markdown resource.

Static HTML pages work with only Python's standard library. JavaScript-rendered
sites need ``--render-js`` and Playwright installed.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = "website-to-markdown/2.0"
IGNORED_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "details",
    "dialog",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
SKIP_EXTENSIONS = {
    ".7z",
    ".avi",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rss",
    ".svg",
    ".tar",
    ".tgz",
    ".webm",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


@dataclass
class PageContent:
    """Markdown-ready content extracted from one page."""

    url: str
    title: str = "Untitled page"
    description: str = ""
    body_markdown: str = ""
    links: list[tuple[str, str]] = field(default_factory=list)
    images: list[tuple[str, str]] = field(default_factory=list)
    script_sources: list[str] = field(default_factory=list)


@dataclass
class LinkCapture:
    """Anchor text collected while parsing an ``<a>`` element."""

    href: str
    text_parts: list[str] = field(default_factory=list)


class MarkdownHTMLParser(HTMLParser):
    """Convert visible HTML body text into simple Markdown.

    This parser intentionally favors completeness over perfect formatting. It
    records text from any visible body element instead of only semantic tags, so
    sites built from nested ``div`` and ``span`` elements still produce useful
    resources.
    """

    def __init__(self, page_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.page_url = page_url
        self.content = PageContent(url=page_url)
        self._body_started = False
        self._ignored_depth = 0
        self._title_depth = 0
        self._title_parts: list[str] = []
        self._parts: list[str] = []
        self._links_in_progress: list[LinkCapture] = []
        self._list_depth = 0
        self._pre_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = {name.lower(): value for name, value in attrs if value is not None}

        if tag == "title":
            self._title_depth += 1
            return

        if tag == "meta":
            self._capture_meta(attrs_map)
            return

        if tag == "script" and "src" in attrs_map:
            self.content.script_sources.append(urljoin(self.page_url, attrs_map["src"]))

        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if tag == "body":
            self._body_started = True
            return

        if not self._body_started or self._ignored_depth:
            return

        if tag in {"ul", "ol"}:
            self._list_depth += 1
            self._newline()
            return

        if tag == "li":
            self._newline()
            self._append(f"{'  ' * max(self._list_depth - 1, 0)}- ")
            return

        heading_match = re.fullmatch(r"h([1-6])", tag)
        if heading_match:
            self._newline()
            self._append(f"{'#' * int(heading_match.group(1))} ")
            return

        if tag == "a" and "href" in attrs_map:
            self._links_in_progress.append(LinkCapture(urljoin(self.page_url, attrs_map["href"])))
            return

        if tag == "img":
            self._capture_image(attrs_map)
            return

        if tag == "br":
            self._newline()
            return

        if tag == "hr":
            self._newline()
            self._append("---")
            self._newline()
            return

        if tag == "pre":
            self._pre_depth += 1
            self._newline()
            self._append("```")
            self._newline()
            return

        if tag in {"td", "th"}:
            self._append(" | ")
            return

        if tag in BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)
            title = clean_text("".join(self._title_parts))
            if title:
                self.content.title = title
            return

        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return

        if tag == "body":
            self._body_started = False
            return

        if not self._body_started or self._ignored_depth:
            return

        if tag == "a" and self._links_in_progress:
            link = self._links_in_progress.pop()
            text = clean_text("".join(link.text_parts)) or link.href
            self.content.links.append((text, link.href))
            return

        if tag in {"ul", "ol"}:
            self._list_depth = max(0, self._list_depth - 1)
            self._newline()
            return

        if tag == "pre":
            self._newline()
            self._append("```")
            self._newline()
            self._pre_depth = max(0, self._pre_depth - 1)
            return

        if tag in {"td", "th"}:
            self._append(" | ")
            return

        if tag in BLOCK_TAGS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self._title_parts.append(data)

        if not self._body_started or self._ignored_depth:
            return

        self._append(data)
        for link in self._links_in_progress:
            link.text_parts.append(data)

    def close(self) -> None:
        super().close()
        self.content.body_markdown = normalize_markdown("".join(self._parts))

    def _capture_meta(self, attrs: dict[str, str]) -> None:
        key = attrs.get("name", attrs.get("property", "")).lower()
        value = clean_text(attrs.get("content", ""))
        if key in {"description", "og:description", "twitter:description"} and value:
            if not self.content.description:
                self.content.description = value
        elif key in {"og:title", "twitter:title"} and value and self.content.title == "Untitled page":
            self.content.title = value

    def _capture_image(self, attrs: dict[str, str]) -> None:
        src = attrs.get("src")
        if not src:
            return

        href = urljoin(self.page_url, src)
        alt = clean_text(attrs.get("alt", attrs.get("title", "Image"))) or "Image"
        self.content.images.append((alt, href))
        self._append(f"![{markdown_escape(alt)}]({href})")

    def _append(self, value: str) -> None:
        self._parts.append(value)

    def _newline(self) -> None:
        self._parts.append("\n")


class StaticFetcher:
    """Fetch static HTML with urllib."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> StaticFetcher:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def fetch(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=self.timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError(f"Skipped non-HTML content type: {content_type or 'unknown'}")

            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")


class PlaywrightFetcher:
    """Fetch HTML after browser rendering for JavaScript-heavy sites."""

    def __init__(self, timeout: float, render_wait: float) -> None:
        self.timeout = timeout
        self.render_wait = render_wait
        self._playwright = None
        self._browser = None
        self._context = None

    def __enter__(self) -> PlaywrightFetcher:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as error:
            raise RuntimeError(
                "--render-js requires Playwright. Install it with: "
                "python3 -m pip install playwright && python3 -m playwright install chromium"
            ) from error

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._context = self._browser.new_context(user_agent=USER_AGENT)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def fetch(self, url: str) -> str:
        if self._context is None:
            raise RuntimeError("Playwright browser context is not ready")

        page = self._context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=int(self.timeout * 1000))
            if self.render_wait > 0:
                page.wait_for_timeout(int(self.render_wait * 1000))
            return page.content()
        finally:
            page.close()


def clean_text(value: str) -> str:
    """Decode entities and collapse whitespace to one line."""

    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def normalize_markdown(value: str) -> str:
    """Collapse noisy HTML whitespace while preserving Markdown paragraphs."""

    value = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    normalized_lines: list[str] = []

    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        normalized_lines.append(line)

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines).strip()


def normalize_url(url: str) -> str:
    """Return an absolute URL without fragments or trailing slash noise."""

    if not urlparse(url).scheme:
        url = f"https://{url}"

    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    normalized = parsed._replace(path=parsed.path or "/").geturl()
    return normalized.rstrip("/") if parsed.path == "/" else normalized


def is_crawlable_url(url: str, start_domain: str, include_external: bool) -> bool:
    """Decide whether a discovered URL should be added to the crawl queue."""

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    if not include_external and parsed.netloc != start_domain:
        return False

    return not any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS)


def parse_page(url: str, html_text: str) -> PageContent:
    """Convert raw or rendered HTML into Markdown-ready page content."""

    parser = MarkdownHTMLParser(url)
    parser.feed(html_text)
    parser.close()
    return parser.content


def crawl(
    start_url: str,
    max_pages: int,
    include_external: bool,
    timeout: float,
    delay: float,
    render_js: bool,
    render_wait: float,
) -> list[PageContent]:
    """Breadth-first crawl from a starting page and return parsed pages."""

    start_url = normalize_url(start_url)
    start_domain = urlparse(start_url).netloc
    queue: deque[str] = deque([start_url])
    queued = {start_url}
    visited: set[str] = set()
    pages: list[PageContent] = []
    fetcher_class = PlaywrightFetcher if render_js else StaticFetcher

    with fetcher_class(timeout, render_wait) if render_js else fetcher_class(timeout) as fetcher:
        while queue and (max_pages == 0 or len(pages) < max_pages):
            current_url = queue.popleft()
            if current_url in visited:
                continue

            visited.add(current_url)
            print(f"Fetching {current_url}", file=sys.stderr)

            try:
                html_text = fetcher.fetch(current_url)
            except Exception as error:  # Keep crawling when one page fails to load.
                print(f"Skipping {current_url}: {error}", file=sys.stderr)
                continue

            page = parse_page(current_url, html_text)

            pages.append(page)

            for _text, href in page.links:
                next_url = normalize_url(href)
                if next_url not in queued and is_crawlable_url(next_url, start_domain, include_external):
                    queue.append(next_url)
                    queued.add(next_url)

            if delay > 0 and queue and (max_pages == 0 or len(pages) < max_pages):
                time.sleep(delay)

    return pages


def markdown_escape(text: str) -> str:
    """Escape Markdown delimiters in extracted labels."""

    return text.replace("[", "\\[").replace("]", "\\]").replace("|", "\\|")


def unique_pairs(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Return pairs in original order without duplicates."""

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def page_to_markdown(page: PageContent) -> str:
    """Render one parsed page as Markdown."""

    lines = [f"# {page.title}", "", f"Source: {page.url}", ""]

    if page.description:
        lines.extend(["## Description", "", page.description, ""])

    lines.extend(["## Content", ""])
    if page.body_markdown:
        lines.extend([page.body_markdown, ""])
    else:
        lines.extend(
            [
                "No visible body text was found in the fetched HTML.",
                "If this page is rendered by JavaScript, rerun with `--render-js`.",
                "",
            ]
        )

    links = unique_pairs(page.links)
    if links:
        lines.extend(["## Links", ""])
        for text, href in links:
            lines.append(f"- [{markdown_escape(text)}]({href})")
        lines.append("")

    images = unique_pairs(page.images)
    if images:
        lines.extend(["## Images", ""])
        for alt, href in images:
            lines.append(f"- [{markdown_escape(alt)}]({href})")
        lines.append("")

    if not page.body_markdown and page.script_sources:
        lines.extend(["## JavaScript Sources", ""])
        for src in page.script_sources:
            lines.append(f"- {src}")
        lines.append("")

    return "\n".join(lines).strip()


def pages_to_markdown(pages: Iterable[PageContent], start_url: str, render_js: bool) -> str:
    """Render the full crawl result as a single Markdown resource."""

    pages = list(pages)
    lines = [
        "# Website Resource",
        "",
        f"Start URL: {start_url}",
        f"Pages captured: {len(pages)}",
        f"JavaScript rendering: {'enabled' if render_js else 'disabled'}",
        "",
    ]

    for index, page in enumerate(pages, start=1):
        if index > 1:
            lines.extend(["", "---", ""])
        lines.append(page_to_markdown(page))

    return "\n".join(lines).strip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Crawl a website and convert discovered HTML pages into one Markdown file."
    )
    parser.add_argument("url", help="Starting website URL, for example https://docs.python.org/3/")
    parser.add_argument(
        "-o",
        "--output",
        default="website_resource.md",
        help="Markdown output file path. Defaults to website_resource.md.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=25,
        help="Maximum number of pages to capture. Use 0 for no limit. Defaults to 25.",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also crawl links outside the starting URL's domain. Disabled by default.",
    )
    parser.add_argument(
        "--render-js",
        action="store_true",
        help="Render pages in Chromium before extracting content. Requires Playwright.",
    )
    parser.add_argument(
        "--render-wait",
        type=float,
        default=2.0,
        help="Seconds to wait after DOM load when --render-js is used. Defaults to 2.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds. Defaults to 10.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between requests in seconds. Defaults to 0.2.",
    )
    return parser


def main() -> int:
    """Run the crawler from the command line."""

    args = build_parser().parse_args()
    if args.max_pages < 0:
        print("--max-pages must be 0 or greater", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be greater than 0", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("--delay must be 0 or greater", file=sys.stderr)
        return 2
    if args.render_wait < 0:
        print("--render-wait must be 0 or greater", file=sys.stderr)
        return 2

    try:
        pages = crawl(
            start_url=args.url,
            max_pages=args.max_pages,
            include_external=args.include_external,
            timeout=args.timeout,
            delay=args.delay,
            render_js=args.render_js,
            render_wait=args.render_wait,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    markdown = pages_to_markdown(pages, normalize_url(args.url), args.render_js)
    Path(args.output).write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(pages)} page(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
