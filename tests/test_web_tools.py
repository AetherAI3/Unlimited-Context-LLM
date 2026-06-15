# aether-context (Unlimited Context)
# Copyright (c) 2026 Aether AI
# SPDX-License-Identifier: Apache-2.0
"""Web tools — SSRF host-allow guard, html-to-text extraction, and the
DuckDuckGo-lite result parser. Network is never touched: the SSRF refusal path
is asserted against loopback/private literals, and the happy path injects HTML
into the extractor/parser directly (the production guard is NEVER weakened)."""
from __future__ import annotations

from aether_agent import web


# --- is_safe_url: scheme + SSRF host-allow check --------------------------
def test_is_safe_url_refuses_non_http_schemes():
    for url in ("file:///etc/passwd", "ftp://example.com/x", "gopher://h/", "data:text/plain,x"):
        ok = web.is_safe_url(url)
        assert ok is not True, f"{url} should be refused"


def test_is_safe_url_refuses_loopback_and_localhost():
    for url in (
        "http://127.0.0.1/",
        "http://127.0.0.1:8080/path",
        "https://localhost/x",
        "http://[::1]/",
    ):
        ok = web.is_safe_url(url)
        assert ok is not True, f"{url} should be refused (loopback)"


def test_is_safe_url_refuses_private_ranges():
    for url in (
        "http://10.0.0.5/",
        "http://10.255.255.255/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://172.31.255.255/",
        "http://169.254.169.254/",  # cloud metadata link-local
    ):
        ok = web.is_safe_url(url)
        assert ok is not True, f"{url} should be refused (private)"


def test_is_safe_url_returns_reason_string_on_refusal():
    # A falsy/reason value (not True) is returned so the caller can surface why.
    res = web.is_safe_url("http://127.0.0.1/")
    assert res is not True
    assert isinstance(res, str) and res  # a non-empty reason


def test_is_safe_url_allows_public_host(monkeypatch):
    # Inject the resolver so we never hit the network. A public IP must pass.
    monkeypatch.setattr(web, "_resolve_ips", lambda host: ["93.184.216.34"])  # example.com
    assert web.is_safe_url("https://example.com/page") is True


def test_is_safe_url_blocks_dns_rebind_to_private(monkeypatch):
    # Even a public-looking host that RESOLVES to a private IP must be refused.
    monkeypatch.setattr(web, "_resolve_ips", lambda host: ["10.0.0.1"])
    res = web.is_safe_url("https://sneaky.example.com/")
    assert res is not True


# --- html -> text extraction ----------------------------------------------
def test_extract_text_strips_script_and_style_and_tags():
    html = (
        "<html><head><style>.x{color:red}</style>"
        "<script>alert('xss');var a=1<2;</script></head>"
        "<body><h1>Title</h1><p>Hello <b>world</b>.</p>"
        "<script>steal()</script></body></html>"
    )
    text = web._extract_text(html)
    assert "Title" in text
    assert "Hello" in text and "world" in text
    assert "alert" not in text
    assert "steal" not in text
    assert "color:red" not in text
    assert "<" not in text and ">" not in text


def test_extract_text_caps_length():
    html = "<p>" + ("a" * 50000) + "</p>"
    text = web._extract_text(html, max_chars=8000)
    assert len(text) <= 8000


# --- web_fetch refusal path (no network) ----------------------------------
def test_web_fetch_refuses_loopback_without_network():
    out = web.web_fetch("http://127.0.0.1:9/")
    assert out.startswith("[web_fetch refused:")


def test_web_fetch_refuses_file_scheme():
    out = web.web_fetch("file:///etc/passwd")
    assert out.startswith("[web_fetch refused:")


# --- DuckDuckGo-lite result parsing into the line format ------------------
def test_web_search_parses_ddg_lite_fixture_into_lines(monkeypatch):
    # A trimmed DDG-lite results table: each result is an anchor with class
    # 'result-link' plus a snippet cell with class 'result-snippet'.
    fixture = """
    <html><body><table>
      <tr><td><a class="result-link" href="https://a.example/one">First Result</a></td></tr>
      <tr><td class="result-snippet">Snippet about the first result.</td></tr>
      <tr><td><a class="result-link" href="https://b.example/two">Second Result</a></td></tr>
      <tr><td class="result-snippet">Snippet about the second result.</td></tr>
      <tr><td><a class="result-link" href="https://c.example/three">Third Result</a></td></tr>
      <tr><td class="result-snippet">Third snippet.</td></tr>
    </table></body></html>
    """
    # Inject the fetcher so no network is touched.
    monkeypatch.setattr(web, "_http_post", lambda url, data, **kw: fixture)
    out = web.web_search("anything", limit=2)
    # Limited to 2 results.
    assert "First Result" in out
    assert "https://a.example/one" in out
    assert "Snippet about the first result." in out
    assert "Second Result" in out
    assert "Third Result" not in out  # capped at limit=2
    # Numbered line format "N. <title>".
    assert out.lstrip().startswith("1.")
    assert "2. Second Result" in out


def test_web_search_network_error_returns_clean_string(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(web, "_http_post", boom)
    out = web.web_search("q")
    assert out.startswith("[web_search error:")


def test_web_search_no_results_message(monkeypatch):
    monkeypatch.setattr(web, "_http_post", lambda *a, **k: "<html><body>nothing</body></html>")
    out = web.web_search("q")
    assert isinstance(out, str) and out  # never raises, always a string


# --- happy-path fetch via a real loopback stub, guard bypassed by injection -
def test_web_fetch_happy_path_extracts_text(monkeypatch):
    # Bind a real http.server to 127.0.0.1 and let web_fetch reach it via the
    # HOSTNAME 'localhost' (not a literal IP, so the guard consults the resolver).
    # We inject _resolve_ips to report a PUBLIC ip, so the production guard ALLOWS
    # the request; the real socket still connects to localhost -> the loopback
    # stub. The guard logic is never weakened — only the address it sees during
    # resolution is swapped — proving the extract-on-success path end to end.
    import http.server
    import threading

    body = b"<html><body><h1>Hi</h1><p>fetched body text</p><script>x()</script></body></html>"

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        # Hostname (not literal IP) so the guard resolves it; injected resolver
        # reports a public IP so the guard allows it. urllib still dials the
        # hostname, which resolves to 127.0.0.1 -> the stub.
        monkeypatch.setattr(web, "_resolve_ips", lambda host: ["93.184.216.34"])
        out = web.web_fetch(f"http://localhost:{port}/")
        assert "fetched body text" in out
        assert "Hi" in out
        assert "x()" not in out  # script stripped
    finally:
        srv.shutdown()
        srv.server_close()


def test_web_fetch_refuses_redirect_to_internal_host(monkeypatch):
    # SSRF redirect bypass: an allowed (public) host MUST NOT be able to 3xx-bounce
    # the fetch to an internal/loopback target. We stand up two loopback servers:
    # a redirector that 302s to a 'secret' internal page, and the secret page. The
    # guard is told (via _resolve_ips) the INITIAL host is public, so it allows the
    # first hop; the redirect target is loopback and MUST be refused per-hop. The
    # internal body must never reach the caller, and web_fetch must not raise.
    import http.server
    import threading

    secret = b"<html><body><p>INTERNAL SECRET PAGE</p></body></html>"

    class Secret(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(secret)))
            self.end_headers()
            self.wfile.write(secret)

        def log_message(self, *a):
            pass

    ssrv = http.server.HTTPServer(("127.0.0.1", 0), Secret)
    sport = ssrv.server_address[1]
    threading.Thread(target=ssrv.serve_forever, daemon=True).start()

    class Redir(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sport}/secret")
            self.end_headers()

        def log_message(self, *a):
            pass

    rsrv = http.server.HTTPServer(("127.0.0.1", 0), Redir)
    rport = rsrv.server_address[1]
    threading.Thread(target=rsrv.serve_forever, daemon=True).start()

    try:
        # Initial host resolves (spoofed) to PUBLIC -> guard allows hop 1. The
        # redirect target (127.0.0.1) is internal -> the redirect handler refuses.
        monkeypatch.setattr(web, "_resolve_ips", lambda host: ["93.184.216.34"])
        out = web.web_fetch(f"http://localhost:{rport}/start")
        assert "INTERNAL SECRET PAGE" not in out, "redirect to internal host leaked"
        assert out.startswith("[web_fetch error"), f"expected a clean error, got {out!r}"
    finally:
        ssrv.shutdown()
        ssrv.server_close()
        rsrv.shutdown()
        rsrv.server_close()
