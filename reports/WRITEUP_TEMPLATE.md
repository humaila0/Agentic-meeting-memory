# Agentic AI Meeting Memory — Technical Write-up

## 1. Problem and approach

The project processes 138 AMI meeting transcripts and answers questions using evidence retrieved from the meetings. Transcript content is parsed into speaker turns, with meeting and speaker identifiers retained so that retrieved passages can be traced back to their sources.

The turns are indexed for semantic retrieval. When a question is received, the agent searches relevant team meetings, gathers supporting passages, and generates an answer with transcript citations.

The approach combines semantic retrieval with deterministic validation. The language model helps interpret questions and synthesize answers, while Python handles transcript processing, metadata filtering, citation matching, and output validation.

## 2. Architecture and tool choices

- **Python:** Used for deterministic transcript parsing, filtering, retrieval orchestration, citation checks, and output validation. These operations are kept in code because they require repeatable and inspectable behaviour.
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Generates local text embeddings, avoiding separate embedding API costs. Semantic retrieval can miss exact numbers, uncommon terms, or specific wording. The agent therefore searches multiple question aspects and inspects retrieved evidence.
- **ChromaDB:** Provides a persistent local vector store for transcript chunks and metadata. The current implementation is a single-machine prototype. A production system handling 10,000 or more meetings would need scalable storage, metadata indexes, backups, monitoring, and incremental ingestion.
- **Gemini API:** Used for language-based question interpretation and answer synthesis. Model and free-tier quota availability can change, and generated answers can contain unsupported claims. Python therefore checks citations and quoted evidence rather than relying only on the model's confidence.

## 3. Agent behaviour

The agent follows a multi-step retrieval and answer process:

1. Interpret the question and identify the relevant team and question aspects.
2. Search the team-scoped transcript index for evidence.
3. Check whether retrieved passages cover the different aspects of the question and whether sufficient meeting coverage has been found.
4. Broaden or separate searches when evidence coverage is weak.
5. Ask the language model to draft an answer using the retrieved evidence.
6. Map cited evidence identifiers back to transcript turns and validate citation requirements, including exact quote matching and meeting coverage.
7. Flag answers that fail validation instead of treating them as fully verified.

This separation makes retrieval and validation more repeatable. It does not guarantee that every answer is factually correct; human review remains important when evidence is ambiguous or incomplete.

## 4. Temporal memory

The project preserves meeting identifiers and chronological meeting information in transcript records and its temporal-memory output. This allows evidence from earlier and later meetings to be compared.

A decision should be described as changed or superseded only when transcript evidence supports that conclusion. Similarity between two passages is not, by itself, proof that one decision replaced another.

The temporal memory output is a supporting data structure, not an independent guarantee that every implicit reference or decision change has been resolved correctly. More extensive testing with manually verified decision histories would be needed to establish that capability reliably.

## 5. Evaluation

The available citation-check results are:

| Evaluation set          | Questions checked | Passed citation checks | Citation issues |
| ----------------------- | ----------------: | ---------------------: | --------------: |
| Main question set       |                25 |                     20 |               5 |
| Additional question set |                10 |                      8 |               2 |
| **Combined**            |            **35** |           **28 (80%)** |           **7** |

The 80% figure is a **citation-validation pass rate, not a factual-accuracy score**. The additional expected-answer file was not available, so a reliable semantic accuracy score against manually prepared gold answers could not be calculated. The manual evidence file contains supporting transcript excerpts, but it is not a complete gold-answer set.

Three observed citation/evidence failures are:

- **EX01:** The answer combined survey findings and later design priorities, but its citation coverage did not include the distinct later meeting evidence. Improvement: retrieve and cite evidence for each part of a multi-part answer separately.
- **EX09:** The answer described both the button-signal mechanism and the later prototype layout, but its citations did not cover the later layout evidence. Improvement: detect missing aspects and retrieve the relevant later transcript turn before finalizing.
- **Q03:** The available citation was too short or incomplete to support the full answer. Improvement: require a sufficiently complete exact quote for each material claim and flag the answer when suitable evidence cannot be found.

These are observed citation-coverage problems, not confirmed factual errors. Confirming factual errors would require complete manual gold answers and transcript review.

## 6. Scale estimate for 1,000 meetings

The supplied assignment corpus contains approximately 880,000 tokens across 138 meetings.

- Average source tokens per meeting: 880,000 ÷ 138 ≈ **6,377**
- Approximate source tokens for 1,000 meetings: 6,377 × 1,000 ≈ **6.38 million**

This is a corpus-size estimate before chunking, metadata, and vector-store overhead.

A reliable ingestion-time or API-cost figure cannot be reported from the available project evidence because measured throughput, run timings, and API usage logs were not available. To produce a measured estimate, time a representative ingestion run, record the number of turns/chunks and embedding throughput, and record any model API calls and token usage. Extrapolate using the same settings while allowing for hardware, batching, retries, and rate limits.

## 7. Limitations and next steps

Current limitations include noisy or incomplete speech transcription, ambiguity in speaker roles, retrieval misses for exact numbers and rare terms, citation coverage gaps, model quota or availability changes, and the need for human review.

The local ChromaDB setup is suitable as a prototype but would need production-grade storage and operations at larger scale.

Recommended next steps:

- Expand the manually verified evaluation set.
- Calculate factual accuracy separately from citation validity.
- Improve retrieval for numeric and rare-term queries.
- Test temporal decision changes with explicit examples.
- Measure ingestion throughput and API usage.
- Add monitoring and human-review workflows.
