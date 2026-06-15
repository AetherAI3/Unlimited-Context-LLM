"""
Network tools for the agent — web_search + web_fetch (stdlib only).

These are the ONLY tools that reach the network, and unlike the file/shell tools
they are NOT path-jailed (there is no workspace to confine to). Instead they are
SSRF-guarded: every request is refused unless its scheme is http(s) AND every IP
the host resolves to is a public, routable address. The guard runs BEFORE any
socket is opened, so a malicious task can never coax the agent into hitting
localhost, the cloud metadata endpoint (169.254.169.254), or an internal RFC-1918
service.

Everything here returns a readable string and NEVER raises — a tool that throws
would abort the brain's turn loop. Errors come back as "[web_* error: ...]".
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Union

# Caps. max_chars bounds what the model sees; MAX_BYTES bounds what we read off
# the socket (defense against a hostile/huge response); TIMEOUT bounds the wait.
MAX_TEXT = 8000
MAX_BYTES = 2_000_000  # ~2 MB read ceiling
TIMEOUT = 15  # seconds
USER_AGENT = "AetherCode/1.0 (+https://aethersystems.net)"

DDG_LITE = "https://lite.duckduckgo.com/lite/"

# is_safe_url returns True (safe) or a reason string (refused).
SafeResult = Union[bool, str]


def _resolve_ips(host: str) -> list[str]:
    """Resolve a host to every IP it points at (v4 + v6). Injected in tests so
    the SSRF guard can be exercised without real DNS. Raises on resolve failure;
    callers treat that as 'cannot verify -> refuse'."""
    infos = socket.getaddrinfo(host, None)
    out: list[str] = []
    for info in infos:
        sockaddr = info[4]
        ip = sockaddr[0]
        if ip not in out:
            out.append(ip)
    return out


def _ip_is_public(ip_str: str) -> bool:
    """True only for globally-routable addresses. Blocks loopback, private
    (10/8, 172.16/12, 192.168/16), link-local (169.254/16, fe80::/10),
    unique-local (fc00::/7), reserved, multicast, and unspecified."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return False
    # ipaddress marks unique-local fc00::/7 as private already, but be explicit.
    return ip.is_global


def is_safe_url(url: str) -> SafeResult:
    """Gate a URL before any network call. Returns True when safe, else a short
    reason string explaining the refusal. Refuses non-http(s) schemes and any
    host that resolves (in whole or in part) to a non-public IP — so a hostname
    that DNS-rebinds to 127.0.0.1 / 10.x / 169.254.x is still blocked."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as e:
        return f"unparseable url ({e})"

    if parts.scheme not in ("http", "https"):
        return f"scheme {parts.scheme or '(none)'!r} not allowed (http/https only)"

    host = parts.hostname
    if not host:
        return "missing host"

    # If the host is already a literal IP, check it directly (no DNS).
    try:
        literal = ipaddress.ip_address(host)
        return True if _ip_is_public(str(literal)) else f"host {host} is not a public address"
    except ValueError:
        pass  # not a literal IP — resolve it below

    try:
        ips = _resolve_ips(host)
    except OSError as e:
        return f"could not resolve {host} ({e})"
    if not ips:
        return f"{host} resolved to no addresses"
    for ip in ips:
        if not _ip_is_public(ip):
            return f"host {host} resolves to non-public address {ip}"
    return True


class _TextExtractor(HTMLParser):
    """Collect visible text, dropping the contents of <script>/<style> entirely."""

    _SKIP = {"script", "style", "noscript", "head", "title"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        # Collapse runs of whitespace per line, drop empty lines.
        lines = [" ".join(ln.split()) for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _extract_text(html: str, max_chars: int = MAX_TEXT) -> str:
    """Strip tags/scripts/styles from HTML and return readable text, capped."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 — malformed html must not raise to the brain
        pass
    return parser.text()[:max_chars]


class _DdgResultParser(HTMLParser):
    """Parse DuckDuckGo-lite result rows into (title, url, snippet) tuples.

    DDG-lite renders each result as an <a class="result-link" href=...>title</a>
    followed by a <td class="result-snippet">snippet</td>. We pair anchors with
    the next snippet cell in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._in_link = False
        self._cur_url = ""
        self._cur_title: list[str] = []
        self._in_snippet = False
        self._cur_snippet: list[str] = []
        self._pending: dict | None = None

    @staticmethod
    def _has_class(attrs, cls: str) -> bool:
        for k, v in attrs:
            if k == "class" and v and cls in v.split():
                return True
        return False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a" and self._has_class(attrs, "result-link"):
            href = ""
            for k, v in attrs:
                if k == "href":
                    href = v or ""
            self._in_link = True
            self._cur_url = href
            self._cur_title = []
        elif tag == "td" and self._has_class(attrs, "result-snippet"):
            self._in_snippet = True
            self._cur_snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            title = " ".join("".join(self._cur_title).split())
            # Flush any prior pending result that never got a snippet.
            if self._pending is not None:
                self.results.append(self._pending)
            self._pending = {"title": title, "url": self._cur_url, "snippet": ""}
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join("".join(self._cur_snippet).split())
            if self._pending is not None:
                self._pending["snippet"] = snippet
                self.results.append(self._pending)
                self._pending = None

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._cur_title.append(data)
        elif self._in_snippet:
            self._cur_snippet.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()
        if self._pending is not None:
            self.results.append(self._pending)
            self._pending = None


def _parse_ddg_results(html: str) -> list[dict]:
    parser = _DdgResultParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    return [r for r in parser.results if r.get("title")]


def _http_post(url: str, data: dict, *, timeout: int = TIMEOUT) -> str:
    """POST form-encoded data and return the decoded body. Injected in tests."""
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/html",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — scheme is fixed https
        raw = resp.read(MAX_BYTES)
    return raw.decode("utf-8", errors="replace")


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-run the SSRF guard on EVERY redirect hop.

    Without this, a public host that passes is_safe_url could 3xx-redirect the
    request to http://169.254.169.254/ (cloud metadata) or an RFC-1918 service,
    and urllib would silently follow it — defeating the very guard web_fetch
    exists to enforce. We refuse such a redirect by raising, which urllib turns
    into an HTTPError the caller surfaces as a clean '[web_fetch error: ...]'."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        safe = is_safe_url(newurl)
        if safe is not True:
            raise urllib.error.HTTPError(newurl, code, f"refused redirect: {safe}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# A module-level opener that enforces the SSRF guard across redirects. Built once;
# the handler is stateless. Tests can still inject _resolve_ips to steer the guard.
_GUARDED_OPENER = urllib.request.build_opener(_GuardedRedirectHandler())


def _http_get(url: str, *, timeout: int = TIMEOUT) -> str:
    """GET a URL (caller MUST have passed is_safe_url first) and return the body,
    capped at MAX_BYTES. Every redirect hop is re-validated by the SSRF guard
    (via _GuardedRedirectHandler) so a public host cannot bounce us to an
    internal/metadata address."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with _GUARDED_OPENER.open(req, timeout=timeout) as resp:  # noqa: S310 — guarded + per-hop re-checked
        raw = resp.read(MAX_BYTES)
        charset = "utf-8"
        ctype = resp.headers.get("Content-Type", "")
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    return raw.decode(charset, errors="replace")


def web_search(query: str, limit: int = 5) -> str:
    """Search the web via DuckDuckGo-lite and return up to `limit` results as
    readable text. Never raises — network/parse errors come back as a string."""
    if not query or not query.strip():
        return "[web_search error: empty query]"
    try:
        limit = max(1, min(int(limit), 25))
    except (TypeError, ValueError):
        limit = 5
    try:
        html = _http_post(DDG_LITE, {"q": query})
    except Exception as e:  # noqa: BLE001 — surface, never raise
        return f"[web_search error: {e}]"

    results = _parse_ddg_results(html)[:limit]
    if not results:
        return f"[web_search: no results for {query!r}]"

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)


def web_fetch(url: str, max_chars: int = MAX_TEXT) -> str:
    """Fetch a URL and return its readable text (tags/scripts stripped), capped.

    SSRF-guarded: refuses non-http(s) and any host resolving to a non-public IP.
    On refusal returns "[web_fetch refused: <reason>]"; on error
    "[web_fetch error: ...]". Never raises."""
    safe = is_safe_url(url)
    if safe is not True:
        return f"[web_fetch refused: {safe}]"
    try:
        cap = int(max_chars) if max_chars else MAX_TEXT
    except (TypeError, ValueError):
        cap = MAX_TEXT
    try:
        html = _http_get(url)
    except Exception as e:  # noqa: BLE001 — surface, never raise
        return f"[web_fetch error: {e}]"
    return _extract_text(html, max_chars=cap)
