# Contributing

**Control before action. Evidence after.**

1. State the invariant (failing test or receipt schema assertion) before changing harness behavior.
2. Update receipts or explicitly leave claims as `NOT_CLAIMED` / `BLOCKED` / `UNAVAILABLE`.
3. Canonical surfaces: product https://a-11-oy.com · proof https://a11oy.net.

## Branch model

- Branch from default as `szl/<change>` or `rosa/<change>`.
- Conventional Commits; one concern per PR.
- Land via pull request only.

## Measurement

A PR that adds a retrieval result must either admit a hash-chain-verified `MEASURED` receipt or keep the public results payload unchanged. Synthetic fixtures stay labeled `LOCAL_FIXTURE_VERIFIER_CONTROLS`.
