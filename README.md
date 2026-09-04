# Retrieval Bench

Measured retrieval evidence for the SZL stack — including the **multivector** lane.

## What this is

The honest companion to every retrieval claim: dense, sparse, and multivector (ColBERT-style) results with the corpus, machine, and date attached. Leaderboard-shaped claims without measurements do not appear here.

## Guarantees

- **Multivector support** — per-token (late-interaction) embeddings with MaxSim scoring: ColBERT-quality ranking semantics without a cross-encoder round-trip at query time.
- **Dense and sparse, too** — ANN (HNSW) and BM25 baselines measured under the same harness.
- **Honest labels** — every published number carries corpus, hardware, and date; unmeasured claims are marked absent.
- **Fail-closed display** — the public surface renders only verified results.

## Public surface

The consolidated public bench lives at [betterwithage/szl-bench-suite](https://huggingface.co/spaces/betterwithage/szl-bench-suite) (Retrieval Bench tab) — one evidence surface for engine, retrieval, and quantization claims.

Hardware truth is sourced from the published runtime witness ([szl-holdings/lutar-runtime-witness](https://github.com/szl-holdings/lutar-runtime-witness)), whose verifier recomputes every digest from source and fails closed on drift.

**Division of labor:** this repo owns retrieval receipts and their fail-closed verifier. The single Space publisher lives in [szl-holdings/frontier-bench](https://github.com/szl-holdings/frontier-bench), which verifies and combines all three planes before one atomic Space commit. The measurement harness that produces receipted retrieval runs lives in [szl-holdings/szl-retrieval-bench](https://github.com/szl-holdings/szl-retrieval-bench); the Wave 1 consolidated bakeoff report is [szl-holdings/szl-wave1-report](https://github.com/szl-holdings/szl-wave1-report).

## Status

Foundation (2026-09-03): honest-results contract, bench schema (dense / sparse / multivector), and verifier in place. Measured results land here from the dedicated GPU node as runs complete.
