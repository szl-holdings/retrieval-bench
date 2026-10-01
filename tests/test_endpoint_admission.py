"""Remote endpoints are admitted only from the operator allowlist (CodeQL py/full-ssrf)."""
import os

import pytest

from retrieval import fusion


@pytest.fixture
def allowlist(monkeypatch):
    monkeypatch.setenv(fusion.ENDPOINT_ALLOWLIST_ENV, "https://embed.internal:8443, http://localhost:8080/")
    yield


def test_no_allowlist_means_every_endpoint_is_refused(monkeypatch):
    monkeypatch.delenv(fusion.ENDPOINT_ALLOWLIST_ENV, raising=False)
    assert fusion.admit_endpoint("https://embed.internal:8443/v1/embeddings") is None
    assert fusion.DenseRetriever("https://embed.internal:8443/v1/embeddings", "m").is_configured() is False
    assert fusion.CrossEncoderReranker("https://embed.internal:8443/rerank", "m").is_configured() is False


def test_allowlisted_origin_is_rebuilt_from_configuration(allowlist):
    admitted = fusion.admit_endpoint("https://EMBED.internal:8443/v1/embeddings?x=1")
    assert admitted == "https://embed.internal:8443/v1/embeddings?x=1"
    assert fusion.admit_endpoint("http://localhost:8080/rerank") == "http://localhost:8080/rerank"


@pytest.mark.parametrize("url", [
    "https://evil.example/v1/embeddings",
    "https://embed.internal/v1/embeddings",          # port differs from the allowlisted origin
    "https://user:pw@embed.internal:8443/v1",         # credentials
    "https://embed.internal:8443/v1#frag",            # fragment
    "file:///etc/passwd",
    "",
    None,
])
def test_foreign_or_malformed_endpoints_are_refused(allowlist, url):
    assert fusion.admit_endpoint(url) is None


def test_post_json_refuses_unadmitted_url(monkeypatch):
    monkeypatch.delenv(fusion.ENDPOINT_ALLOWLIST_ENV, raising=False)
    with pytest.raises(fusion.EndpointUnavailable):
        fusion._post_json("https://evil.example/v1/embeddings", {"input": ["x"]})
