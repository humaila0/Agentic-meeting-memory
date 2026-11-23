# Agentic AI Meeting Memory

A local-first Agentic AI prototype for the AMI meeting transcript build test. It processes 138 meeting transcripts, indexes speaker turns using local embeddings, stores searchable data in ChromaDB, and uses Gemini to retrieve and synthesize evidence across meetings.

The system also builds temporal memory and generates answers with transcript citations. Python validation checks whether cited quotations match the original transcript turns.

## Important

- Never commit `.env` or expose your Gemini API key.
- The default Gemini model in `.env.example` must be checked against the models currently available to your API key in Google AI Studio. Change `GEMINI_MODEL` if necessary.
- Generated `answers.json` and the TS3012 brief are drafts until manually reviewed. Citation validation does not guarantee factual correctness.
- The TS3012 pre-final brief must use meetings **a, b, and c only**. Meeting d is the final meeting and must not be used for a pre-final brief.
- The original assignment transcripts and questions are not included in this repository. Obtain the dataset separately.

## 1. Requirements

- Windows 10 or 11
- Python 3.10 or 3.11
- Internet connection for initial package and embedding-model downloads
- Gemini API key from Google AI Studio

Transcript parsing, embeddings, and vector storage run locally. Gemini is used for language-based planning and answer synthesis.

## 2. Dataset setup

Extract the assignment dataset into the following structure:

```text
data/
└── source/
    ├── questions.json
    └── transcripts/
        ├── ES2002a.txt
        ├── ES2002b.txt
        ├── ...
        └── other meeting transcripts
```

Make sure the transcript files are directly inside `data/source/transcripts/`.

Do not create an extra nested `transcripts/transcripts` folder.

**Note:** `data/source/` is excluded from Git through `.gitignore`. You must obtain and place the original assignment data separately before running the project.

## 3. Create the Python environment

Open PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create your environment configuration:

```powershell
Copy-Item .env.example .env
notepad .env
```

Add your Gemini API key and check the model name:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=your_available_model_id
```

Keep `.env` private. Never commit it to GitHub.

## 4. Build searchable meeting memory

Run:

```powershell
python -m src.ingest
```

The ingestion process:

1. Reads the meeting transcript files.
2. Parses transcript content into speaker turns.
3. Preserves meeting and speaker information.
4. Splits unusually long turns where necessary.
5. Generates local embeddings using Sentence Transformers.
6. Stores searchable transcript data and metadata in ChromaDB.

The persistent vector database is stored in `chroma_db/`.

The first run may download the embedding model.

## 5. Build temporal memory

Run:

```powershell
python -m src.build_temporal_memory
```

This creates:

```text
temporal_memory.json
```

The temporal-memory process extracts structured information from meetings, including:

- Decisions and their reported status
- Proposed, accepted, rejected, or changed items
- Action items and owners
- Numbers and relevant details
- Open questions and follow-ups

The process can resume from previously processed meetings if interrupted.

Extracted quotations are checked against original speaker turns. This verifies quotation matching, not whether the model's interpretation is factually correct. Review important decisions and chronology manually.

## 6. Generate the required answers

Run:

```powershell
python -m src.agent
```

This generates:

```text
answers.json
```

The agent retrieves evidence from relevant team meetings, synthesizes answers, and attaches transcript citations.

Answers are expected to contain 1–3 sentences and supporting evidence from multiple meetings where required by the assignment.

Some answers may contain `verification_issues`. Review these manually rather than assuming every generated answer is correct.

## 7. Generate the TS3012 pre-final brief

Run:

```powershell
python -m src.brief
```

This generates:

```text
TS3012_brief.txt
TS3012_brief_audit.json
```

The pre-final brief uses TS3012 meetings **a, b, and c only**. Meeting d is excluded because it is the final meeting.

Review the brief and its audit before using it as a final deliverable.

## 8. Validate generated answers

Run the citation checker on the main answers:

```powershell
python -m src.check_answers answers.json
```

For the additional evaluation answers:

```powershell
python -m src.check_answers evaluation/additional_answers.json
```

The checker reports citation and structural issues. It does not independently establish factual accuracy.

Review flagged answers against their original transcript turns and correct unsupported claims or incomplete citations.

## 9. Additional evaluation

The project includes additional evaluation questions and generated answers under:

```text
evaluation/
├── additional_questions.json
├── additional_answers.json
└── manual_evidence.txt
```

Use these files to review retrieval and citation behaviour.

For a proper factual-accuracy evaluation:

1. Prepare at least 10 original questions.
2. Write manually verified expected answers.
3. Identify exact supporting transcript quotations.
4. Compare generated answers with the expected answers.
5. Record correct, partially correct, and incorrect answers.
6. Report citation validity separately from factual accuracy.
7. Explain three actual failures and how the system could be improved.

Do not present citation-check pass rates as factual-accuracy scores.

## 10. Architecture

```text
AMI Meeting Transcripts
          |
          v
   Transcript Parser
          |
          v
    Speaker Turns
          |
          v
 Local Sentence Embeddings
          |
          v
       ChromaDB
          |
          v
  Temporal Memory Extraction
          |
          v
 Gemini Query Planning and Synthesis
          |
          v
 Python Citation Verification
          |
          v
 JSON Answers and Meeting Brief
```

### Why this architecture?

- Python handles deterministic operations such as parsing, filtering, and validation.
- Sentence Transformers creates local embeddings without a separate embedding API.
- ChromaDB provides persistent semantic search.
- Gemini handles language understanding and answer synthesis.
- Python checks citations against original transcript text to improve traceability.

## 11. Scaling to 1,000 meetings

The assignment corpus contains approximately 880,000 tokens across 138 meetings.

Estimated average:

```text
880,000 / 138 ≈ 6,377 tokens per meeting
```

Estimated source tokens for 1,000 meetings:

```text
6,377 × 1,000 ≈ 6.38 million tokens
```

This is an approximate source-corpus estimate, excluding chunking and metadata overhead.

Actual ingestion time and API cost should be measured from representative runs. Record processing time, embedding throughput, Gemini API calls, and token usage before making a production estimate.

For a larger deployment, consider:

- Incremental transcript ingestion
- Scalable vector storage
- Metadata indexing
- Batch processing
- API rate-limit handling
- Monitoring and backups
- Human review for important decisions

## 12. Limitations and future improvements

Current limitations include:

- Speech transcription errors and noisy meeting content
- Ambiguous speaker identities and roles
- Difficulty retrieving exact numbers or rare terms
- Incomplete evidence coverage for some multi-part questions
- Model-generated interpretations that require human review
- Gemini model availability and API quota limitations
- Local ChromaDB being a prototype rather than a distributed production system

Future improvements include stronger hybrid retrieval, a larger manually verified evaluation set, improved temporal decision tracking, and production-grade monitoring and storage.


