# Retrieval Bench

Retrieval benchmark source, receipt verification, and evidence plumbing for the SZL stack — including the **multivector** lane.

## What this is

This repository owns retrieval-plane benchmark code, retrieval receipts, and their fail-closed verifier. It supports dense, sparse, rerank, hybrid, and multivector (ColBERT-style late-interaction) evaluation paths without turning implementation support or fixture controls into measured production claims.

## Current evidence state

The checked-in retrieval receipt chain currently contains only `receipts/000-genesis.json`, whose status is `BLOCKED`, method states that no run was executed, and metrics are empty. Therefore this repository currently has **zero admitted measured retrieval rows**.

`evidence/eclipse-native-repaired-20260905.json` is a local synthetic verifier-control artifact (`scope: LOCAL_FIXTURE_VERIFIER_CONTROLS`, fixture hardware, dated 2000-01-01). It demonstrates mutation detection for the verifier; it is not benchmark evidence and must not be presented as measured production performance.

## Evidence contract

- **Query-bound run receipts** — in-memory benchmark receipts bind corpus, query set, qrels, model revision, configuration, and result hashes so comparisons cannot silently cross query populations.
- **Multivector support** — per-token late-interaction embeddings with MaxSim scoring are implemented as a separate retrieval family; no measured multivector performance is claimed until an admitted receipt exists.
- **Dense and sparse support** — ANN/dense and BM25 paths are implemented under the same harness; implementation availability is not evidence of measured parity or superiority.
- **Honest labels** — measured claims require admitted receipts with corpus/input provenance, hardware, date, method, and metrics.
- **Fail-closed publication** — `tools/sync_results.py` exports only hash-chain-verified `MEASURED` receipts; non-measured states remain absent from the public results payload.
- **Integrity boundary** — the receipt chain is tamper-evident but unsigned; it proves continuity/integrity, not independent signer or hardware identity.

## Public surface

The consolidated public bench is published by [szl-holdings/frontier-bench](https://github.com/szl-holdings/frontier-bench) to [betterwithage/szl-bench-suite](https://huggingface.co/spaces/betterwithage/szl-bench-suite). Provider publication, runtime readback, and public presentation are separate qualification stages from this repository's source and receipt verification.

**Division of labor:** this repository owns retrieval receipt verification and retrieval-plane source. The measurement harness that produces receipted retrieval runs lives in [szl-holdings/szl-retrieval-bench](https://github.com/szl-holdings/szl-retrieval-bench); the consolidated publisher lives in [szl-holdings/frontier-bench](https://github.com/szl-holdings/frontier-bench); and the Wave 1 consolidated bakeoff report lives in [szl-holdings/szl-wave1-report](https://github.com/szl-holdings/szl-wave1-report).

## Status

Foundation source is present. Current admitted measurement state is **EMPTY / BLOCKED genesis only**. Real benchmark execution and any later promotion remain separate evidence-producing steps.
