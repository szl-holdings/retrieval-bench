# Leader-Study Upgrade Plan - W27 loop-6 - 2026-11-23

Status: PROPOSE-ONLY DRAFT - DO NOT MERGE. Owner-gated: merging requires governing approval and the estate signature-wall owner unlock.

Repo: retrieval-bench - Retrieval Benchmark Plane — sparse BM25, dense, hybrid RRF, cross-encoder rerank lanes with five-state honesty contract (MEASURED/BLOCKED/INVALID/FAILED/PROMOTED), fairness-gated comparisons, SHA-256 evidence receipts, FastAPI + Prometheus surface. Doctrine v11. Apache-2.0.

Frame: OUROBOROS loop-6 leader study. Governing seat: Kimi kimi-k3.1-pro (json_ok, think off). Advisory seats: Astra/Sol/Claude community rails (provenance unverified, advisory only). Leader evidence web-verified 2026-11-23.

## Leader moats to absorb
- BEIR: Zero-shot retrieval evaluation across heterogeneous datasets and tasks, commonly reported with ranking metrics such as nDCG. Moat: Broad, comparable task coverage and an established zero-shot evaluation convention that discourages claims from a single favorable corpus.
- MS MARCO passage ranking: Large-scale passage-ranking benchmark with realistic queries, relevance judgments, and standardized evaluation. Moat: Scale, recognizable query distribution, and community comparability; it tests generalization far beyond a tiny fixture.
- BM25 sparse lexical retrieval: Lexical retrieval using mature tokenization, normalization, and analyzer pipelines, including Elasticsearch-class implementations. Moat: Strong, inexpensive baseline performance plus sensitivity-tested text processing; analyzer choices can materially affect results and reproducibility.
- Dense bi-encoders: Independent query and document encoding for nearest-neighbor retrieval, commonly implemented with sentence-transformers. Moat: Semantic matching beyond exact terms, reusable embeddings, and a large ecosystem of models and ANN infrastructure.
- Reciprocal Rank Fusion (RRF): Rank-only hybrid fusion that combines retriever result lists without directly mixing incompatible scores. Moat: Simple, robust score-scale independence; the k constant and input rankings remain explicit experimental controls.
- Cross-encoder rerankers: Joint query-document scoring to reorder a candidate top-K set, exemplified by Cohere rerank and Voyage rerank-2.5 with long-context support. Moat: Fine-grained relevance modeling after retrieval, with bounded candidate scope and explicit latency/context tradeoffs.
- nDCG-based end-to-end evaluation: Graded, rank-sensitive measurement of retrieval and reranking quality against qrels. Moat: A common quality signal across lanes, provided query sets, qrels, cutoffs, and comparison populations are held fixed.

## Honest gaps (grounded in repo facts)
- The only admitted MEASURED result is sparse BM25 on fixtures/sparse-bm25-seed-v1 with 2 queries and 3 docs; this is far below BEIR or MS MARCO-scale evidence and cannot support broad zero-shot or production claims.
- The receipt chain begins with BLOCKED at receipts/000-genesis.json, and the README explicitly leaves larger corpora unclaimed until their own MEASURED receipts land.
- Dense, hybrid, and rerank lanes are described as implemented but remain without admitted MEASURED receipts, so no measured parity, superiority, fusion benefit, or reranking benefit is established.
- Multivector/ColBERT-style late interaction and MaxSim are implemented, but the README expressly claims no measured multivector performance until an admitted receipt exists.
- Public site/results.json contains only the one verified MEASURED sparse row; there is no public measured evidence yet for the other retrieval families.
- evidence/eclipse-native-repaired-20260905.json is explicitly a LOCAL_FIXTURE_VERIFIER_CONTROLS synthetic artifact with fixture hardware and date 2000-01-01; it demonstrates mutation detection, not benchmark performance.
- The supplied facts establish query-bound receipts, provenance, fairness gating, and hash-chain-verified publication, but do not establish measured analyzer ablations, RRF k sensitivity, reranker top-K/context tradeoffs, or nDCG results on standard corpora.
- The FACTS provide no pushed_at value and no repository test, CI, latency, throughput, or external-corpus result details; none should be inferred from the FastAPI/Prometheus description or implementation claims.

## Dosed upgrade plan
- Dose 1 — preserve the current single MEASURED sparse seed and its public export unchanged; add or refine receipt-schema validation tests for JSON-object receipts/metrics, duplicate-key rejection, query-population binding, and hash-chain verification, labeling all synthetic cases as verifier controls.
- Dose 2 — propose one larger sparse BM25 evaluation on a versioned public corpus such as a BEIR dataset or MS MARCO slice, with fixed analyzers, qrels, query set, hardware, date, metrics, and nDCG cutoff; publish only if the run earns an admitted MEASURED receipt.
- Dose 3 — run the dense bi-encoder lane on that identical admitted query population, binding model revision and ANN configuration; use the fairness gate before any sparse-versus-dense comparison.
- Dose 4 — add hybrid RRF using the already measured sparse and dense inputs, recording candidate depths and k; claim fusion effects only from a new MEASURED receipt on the same population.
- Dose 5 — add cross-encoder reranking over a fixed retrieved top-K, recording reranker revision, context limits, and candidate-set provenance; separately measure quality and operational cost rather than treating long context as a quality result.
- Dose 6 — evaluate multivector late interaction independently under the same harness, then expand corpus coverage gradually; keep every unsupported or failed run BLOCKED, INVALID, or FAILED as applicable and absent from measured public claims.
- Risk note: The main risk is evidence inflation: turning a 2-query/3-document BM25 seed or a synthetic verifier-control artifact into broad benchmark or production claims. Additional risks include cross-query comparisons, unrecorded analyzer/model/ANN changes, RRF k or candidate-depth leakage, reranker gains confounded with base retrieval, and duplicate-key or malformed-receipt admission. Mitigate by small independently receipted steps, identical comparison populations, explicit configuration binding, and continued fail-closed publication.
- Propose-only note: This is a proposal only: do not run benchmarks, alter receipts, change public results, or claim BEIR/MS MARCO, dense, hybrid, rerank, multivector, latency, or production performance without explicit approval and newly admitted MEASURED evidence. Preserve the five-state contract and fail-closed export.

## Advisory notes (unverified rails, non-binding)
- astra: frontier-bench: hardware-specific kernel optimization suites retrieval-bench: industry-standard gold-set evaluation pipelines szl-calibration: probabilistic forecasting uncertainty baselines
- sol: frontier-bench: Hardware-specific kernel optimization profiles retrieval-bench: End-to-end latency-throughput Pareto frontiers szl-calibration: Domain-specific uncertainty-aware calibration baselines
- claude: frontier-bench: Hardware-specific kernel optimization profiles retrieval-bench: Domain-specific synthetic query generation pipelines szl-calibration: Out-of-distribution uncertainty quantification frameworks

## DO NOT MERGE
This plan is a proposal artifact of the 10-pass OUROBOROS loop (loop-6 of 10). No code, no CI, no behavior changes in this file. A reviewer with authority may convert to a work order later.