"""Remote endpoints are admitted only from the operator allowlist (CodeQL py/full-ssrf)."""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from retrieval import fusion


@pytest.fixture
def allowlist(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV,
                       "https://embed.internal:8443/v1/embeddings, http://localhost:8080/rerank")
    yield


def test_no_allowlist_means_every_endpoint_is_refused(monkeypatch):
    monkeypatch.delenv(fusion.ENDPOINT_ALLOWLIST_ENV, raising=False)
    assert fusion.admit_endpoint("https://embed.internal:8443/v1/embeddings") is None
    assert fusion.DenseRetriever("https://embed.internal:8443/v1/embeddings", "m").is_configured() is False
    assert fusion.CrossEncoderReranker("http://localhost:8080/rerank", "m").is_configured() is False


def test_configured_endpoint_is_selected_by_exact_match(allowlist):
    assert fusion.admit_endpoint("https://embed.internal:8443/v1/embeddings") == "https://embed.internal:8443/v1/embeddings"
    assert fusion.admit_endpoint(" http://localhost:8080/rerank ") == "http://localhost:8080/rerank"
    assert fusion.DenseRetriever("https://embed.internal:8443/v1/embeddings", "m").is_configured() is True


@pytest.mark.parametrize("url", [
    "https://evil.example/v1/embeddings",
    "https://embed.internal:8443/v1/embeddings/../../admin",   # same origin, different path: not configured
    "https://embed.internal:8443/v1/embeddings?x=1",            # query not configured
    "https://EMBED.internal:8443/v1/embeddings",                # exact match only
    "https://user:pw@embed.internal:8443/v1/embeddings",
    "file:///etc/passwd",
    "",
    None,
])
def test_anything_not_configured_is_refused(allowlist, url):
    assert fusion.admit_endpoint(url) is None


def test_credentialed_configuration_entries_are_ignored(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, "https://user:pw@embed.internal/v1, ftp://x/y, https://ok.internal/v1")
    assert fusion.allowed_endpoints() == ("https://ok.internal/v1",)


def test_post_json_refuses_unadmitted_url(monkeypatch):
    monkeypatch.delenv(fusion.ENDPOINT_ALLOWLIST_ENV, raising=False)
    with pytest.raises(fusion.EndpointUnavailable):
        fusion._post_json("https://evil.example/v1/embeddings", {"input": ["x"]})


def test_dense_uses_the_exact_configured_endpoint(allowlist, monkeypatch):
    calls = []

    def urlopen(_opener, request, timeout):
        calls.append((request.full_url, json.loads(request.data)))
        return io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": [1, 0]}]}).encode())

    monkeypatch.setattr(fusion.urllib.request.OpenerDirector, "open", urlopen)
    retriever = fusion.DenseRetriever("https://embed.internal:8443/v1/embeddings", "m")
    retriever.index(["d1"], ["document"])
    assert retriever.search("query") == [("d1", 1.0)]
    assert calls == [
        ("https://embed.internal:8443/v1/embeddings", {"model": "m", "input": ["document"]}),
        ("https://embed.internal:8443/v1/embeddings", {"model": "m", "input": ["query"]}),
    ]


@pytest.mark.parametrize("wrapped", [False, True])
def test_rerank_uses_configured_endpoint_and_accepts_tei_results(allowlist, monkeypatch, wrapped):
    results = [{"index": 1, "score": 0.8}, {"index": 0, "score": 0.2}]
    body = {"results": results} if wrapped else results
    calls = []

    def urlopen(_opener, request, timeout):
        calls.append((request.full_url, json.loads(request.data)))
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(fusion.urllib.request.OpenerDirector, "open", urlopen)
    reranker = fusion.CrossEncoderReranker("http://localhost:8080/rerank", "m")
    assert reranker.rerank("query", [("d1", "first"), ("d2", "second")]) == [
        ("d2", 0.8), ("d1", 0.2),
    ]
    assert calls == [("http://localhost:8080/rerank", {
        "model": "m", "query": "query", "texts": ["first", "second"],
    })]


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_admitted_endpoint_cannot_redirect_to_unadmitted_destination(monkeypatch, status):
    destination_calls = []

    class DestinationHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            destination_calls.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_GET

        def log_message(self, *args):
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)
    destination_url = f"http://127.0.0.1:{destination.server_port}/unadmitted"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(status)
            self.send_header("Location", destination_url)
            self.end_headers()

        def log_message(self, *args):
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    endpoint = f"http://127.0.0.1:{source.server_port}/rerank"
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, endpoint)
    threads = [threading.Thread(target=server.serve_forever, daemon=True)
               for server in (source, destination)]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(fusion.EndpointUnavailable):
            fusion._post_json(endpoint, {"query": "must stay at configured endpoint"}, timeout=2)
        assert destination_calls == []
    finally:
        for server in (source, destination):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
