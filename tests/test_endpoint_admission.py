"""Remote endpoints are admitted only from the operator allowlist (CodeQL py/full-ssrf)."""
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
