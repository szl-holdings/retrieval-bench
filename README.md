# Retrieval Bench

<!-- szl:header v1 -->
<!-- Native identity: szl-brand@4309f6fce89179331b0b3ed8c99494e76c5be256; Orbit v2 icon. -->
<p>
  <a href="https://github.com/szl-holdings">
    <img src="https://raw.githubusercontent.com/szl-holdings/szl-brand/4309f6fce89179331b0b3ed8c99494e76c5be256/kit/logos/orbit-v2/suite/szl-icon-32.svg" alt="SZL Holdings — Orbit v2" width="32" height="32" />
  </a>
</p>

**Control before action. Evidence after.**

[Product](https://a-11-oy.com) · [Proof](https://a11oy.net) · [GitHub organization](https://github.com/szl-holdings)
<!-- /szl:header -->

Retrieval benchmark source, receipt verification, and evidence plumbing for the SZL stack — including the **multivector** lane.

## What this is

This repository owns retrieval-plane benchmark code, retrieval receipts, and their fail-closed verifier. It supports dense, sparse, rerank, hybrid, and multivector (ColBERT-style late-interaction) evaluation paths without turning implementation support or fixture controls into measured production claims.

## Current evidence state

The receipt chain starts at `receipts/000-genesis.json` (`BLOCKED`, no run). `receipts/001-sparse-bm25-seed-v1.json` admits the first **MEASURED** sparse BM25 seed (fixture `fixtures/sparse-bm25-seed-v1`, 2 queries / 3 docs). Public `site/results.json` exports that one MEASURED row via `tools/sync_results.py`. Larger corpora and dense/hybrid/rerank lanes remain unclaimed until their own MEASURED receipts land.

`evidence/eclipse-native-repaired-20260905.json` is a local synthetic verifier-control artifact (`scope: LOCAL_FIXTURE_VERIFIER_CONTROLS`, fixture hardware, dated 2000-01-01). It demonstrates mutation detection for the verifier; it is not benchmark evidence and must not be presented as measured production performance.

## Evidence contract

- **Query-bound run receipts** — in-memory benchmark receipts bind corpus, query set, qrels, model revision, configuration, and result hashes so comparisons cannot silently cross query populations.
- **Multivector support** — per-token late-interaction embeddings with MaxSim scoring are implemented as a separate retrieval family; no measured multivector performance is claimed until an admitted receipt exists.
- **Dense and sparse support** — ANN/dense and BM25 paths are implemented under the same harness; implementation availability is not evidence of measured parity or superiority.
- **Honest labels** — measured claims require admitted receipts with corpus/input provenance, hardware, date, method, and metrics.
- **Fail-closed publication** — `tools/sync_results.py` exports only hash-chain-verified `MEASURED` receipts; non-measured states remain absent from the public results payload.
- **Receipt input admission** — receipts and their metrics must be JSON objects. Duplicate keys and non-finite numbers are rejected before publication; a malformed record rejects the whole chain and leaves an existing results file unchanged. These checks validate structure and integrity, not the truth of caller-supplied measurements.
- **Integrity boundary** — the receipt chain is tamper-evident but unsigned; it proves continuity/integrity, not independent signer or hardware identity.

## Public surface

The consolidated public bench publisher, [szl-holdings/frontier-bench](https://github.com/szl-holdings/frontier-bench), targets the Hugging Face Space [SZLHOLDINGS/szl-bench-suite](https://huggingface.co/spaces/SZLHOLDINGS/szl-bench-suite) (Retrieval tab). Provider publication, runtime readback, and public presentation are separate qualification stages from this repository's source and receipt verification.

**Division of labor:** this repository owns retrieval receipt verification and retrieval-plane source. The measurement harness that produces receipted retrieval runs lives in [szl-holdings/szl-retrieval-bench](https://github.com/szl-holdings/szl-retrieval-bench); the consolidated publisher lives in [szl-holdings/frontier-bench](https://github.com/szl-holdings/frontier-bench); and the Wave 1 consolidated bakeoff report lives in [szl-holdings/szl-wave1-report](https://github.com/szl-holdings/szl-wave1-report).

## Status

Foundation source is present. Admitted measurement state is **1 MEASURED sparse BM25 seed row** plus genesis BLOCKED. Dense/hybrid/rerank/multivector promotion remain separate evidence-producing steps.

## Remote endpoint admission

Dense and rerank lanes call an operator-configured embedding or rerank service. The API accepts an `endpoint` / `dense_endpoint` / `rerank_endpoint` field, but a request can only select one of the endpoints the operator has configured in `RETRIEVAL_BENCH_ENDPOINTS` (comma-separated absolute URLs, exact match). With the variable unset, every remote lane returns `BLOCKED` — the plane never issues a request to a caller-chosen URL.

Configure each service's final request URL, including `/v1/embeddings` or `/rerank`. Redirects are rejected before any follow-up request; the selected URL must serve the response directly.

## Cost-aware route proposals (software)

`python/szl_prisca_router` emits Soft Shield mouth proposals (`PROPOSAL_ONLY`). Router win claims stay `NOT_CLAIMED` until cost-aware route traces are MEASURED; the sparse BM25 seed receipt does not promote the router.
