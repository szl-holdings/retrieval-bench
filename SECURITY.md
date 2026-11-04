# Security

Report a vulnerability or leaked credential privately through GitHub: **Security → Report a vulnerability** on this repository. Do not open a public issue for secrets or exploitable defects.

## Scope

This repository owns retrieval-plane benchmark code, receipts, and fail-closed verification. Remote dense/rerank lanes only call operator-allowlisted endpoints (`RETRIEVAL_BENCH_ENDPOINTS`). Caller-chosen URLs are refused.

## Secrets

- Never commit tokens, endpoint credentials, or private corpora with PII.
- Measurement harness credentials stay outside this tree.
- Rotate first if a secret leaks, then open a private advisory.

## Evidence honesty

Admitted `MEASURED` claims require a non-genesis receipt in the hash chain. Genesis `BLOCKED` is not a leaderboard. Do not publish stub metrics.
