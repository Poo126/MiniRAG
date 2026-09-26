# MP2 · Mini-RAG — Ask Your Documents

> *Mini Project 2 · Week 9 deliverable · Phase 2 (Retrieval / RAG)
> ~5-6 hours over 2-3 sittings.*

A standalone end-to-end RAG build. Mirrors the capstone's core retrieval
loop at small scale, on a small public-domain corpus distinct from the
capstone. The point is integration: prove you can compose W6-W9 from a
blank file.

---

## Bundle contents

### For learners

| File | Purpose |
|---|---|
| `learner/MP2_Brief.md` | What to build, how to submit |
| `learner/mp2_rag_starter.py` | Skeleton with 7 TODO functions |
| `learner/corpus/*.txt` | 5 Sherlock Holmes story summaries (public domain) |
| `learner/data/predefined_questions.jsonl` | 2 golden questions to validate against |
| `learner/data/learner_questions.jsonl` | Template for your 3 questions |
| `learner/requirements.txt` | Python dependencies |
| `learner/.env.example` | Environment template (OpenAI + Qdrant) |
| `learner/MP2_FAQ.md` | Anticipated questions + answers |

### For instructors / graders

| File | Purpose |
|---|---|
| `instructor/MP2_Rubric.md` | 4-dimension × 1-4 scoring rubric (generous) |
| `instructor/MP2_Reference_Solution.py` | Complete working implementation (~300 lines) |
| `instructor/MP2_Solution_Document.md` | Reference walkthrough + grading hints |
| `instructor/MP2_Sample_Submission.md` | What 3.8/4.0 looks like (calibration tool) |

---

## At a glance

- **Project:** First end-to-end RAG over a small corpus (load → chunk → embed → Qdrant → retrieve → answer)
- **Corpus:** 5 Sherlock Holmes story summaries (~3-5 KB each, public domain)
- **Stack:** OpenAI (`gpt-4o-mini` + `text-embedding-3-small`) + Qdrant
- **Effort:** ~5-6 hours
- **Spend:** ~$0.05 - $0.20 in OpenAI API calls
- **Submission:** Working script + 5 sample Q&A pairs + 1-page reflection
- **Stretch (optional):** W9 hybrid + cross-encoder rerank

---

## Where this fits in the curriculum

MP2 sits at **Phase 2 mid-point** — paired with W9's first DANGER ZONE
week. Per the curriculum master, it's a standalone integrative build
that mirrors the capstone's core loop at small scale, *deliberately
separate* from the capstone repo so learners have a distinct portfolio
artefact.

| W6-W9 skill | How MP2 exercises it |
|---|---|
| W6 — Basic RAG | The core loop is exactly W6's |
| W7 — Embeddings + Qdrant | Embed the corpus, store in Qdrant, search |
| W8 — Structure-aware chunking | The recommended chunker uses paragraph + section boundaries |
| W9 — Hybrid + rerank | Stretch path adds BM25+RRF and cross-encoder rerank |

**No new concepts. No new tools. Just revision via integration.**

---

## Quick-start (learner path)

### Create a virtual environment

The `.venv` name refers to a virtual-environment **directory**, not a single
file. It stores a private Python interpreter and project-specific packages so
this project does not change packages used by other projects on your computer.
Using a virtual environment also makes it easier to reproduce the required
dependency versions and avoid conflicts between packages.

From the repository root, create and activate the environment with:

```bash
cd learner
python3 -m venv .venv
source .venv/bin/activate
```

After activation, install packages into this environment:

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and QDRANT_URL
source .env

# Fill in the 7 TODO functions in mp2_rag_starter.py
# Rename it to mp2_rag.py when done

python mp2_rag.py ingest        # one-time setup
python mp2_rag.py validate      # check predefined questions
python mp2_rag.py ask           # interactive Q&A
```

When you finish working, leave the environment with:

```bash
deactivate
```

The `.venv/` directory should not be uploaded to GitHub because it is
machine-specific and can be recreated from `requirements.txt`.

### Optional W9 stretch: hybrid + cross-encoder reranking

The learner implementation also supports BM25 + dense reciprocal-rank fusion,
followed by a local `sentence-transformers` cross-encoder reranker. Enable it in
`.env` before running `ask` or `validate`:

```bash
pip install -r requirements.txt
# Use Python 3.11 or 3.12 for the cross-encoder dependency:
pip install -r requirements-stretch.txt
export USE_HYBRID_SEARCH="1"
export USE_RERANK="1"
export RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
python mp2_rag.py ingest
python mp2_rag.py validate
```

The first reranked query downloads the cross-encoder model from Hugging Face.
The cross-encoder dependency is kept separate because PyTorch wheels may not be
available for every Python version or platform.
The basic dense-only path remains the default.

Then write your 3 questions in `data/learner_questions.jsonl` and your
reflection in `mp2_reflection.md`.

---

## Quick-start (instructor path)

```bash
cd instructor/
# Reference solution is identical structure to the starter,
# with all 7 TODOs filled in.

python MP2_Reference_Solution.py ingest
python MP2_Reference_Solution.py validate
python MP2_Reference_Solution.py ask
```

Expected output: both predefined questions ✓ on source-match.

For grading, read:

1. `MP2_Rubric.md` — the 4-dimension scoring
2. `MP2_Solution_Document.md` — what to look for in submissions
3. `MP2_Sample_Submission.md` — a calibrated 3.8/4.0 example

---

## Corpus provenance

The 5 stories in `learner/corpus/` are condensed summaries (in our own
words) of Arthur Conan Doyle's *The Adventures of Sherlock Holmes*,
published 1892 (public domain in all jurisdictions). The plots,
characters, and named locations are public-domain ideas. The narration
is original to this curriculum — we do not reproduce Doyle's prose.

Files:

- `01_red_headed_league.txt` — The Red-Headed League
- `02_speckled_band.txt` — The Adventure of the Speckled Band
- `03_blue_carbuncle.txt` — The Adventure of the Blue Carbuncle
- `04_engineers_thumb.txt` — The Adventure of the Engineer's Thumb
- `05_scandal_in_bohemia.txt` — A Scandal in Bohemia

Learners are welcome to substitute their own corpus (the brief explains
how). The Sherlock corpus is the default for cohort consistency in
grading.

---

## Cohort calibration expectations

After ~5 submissions, expect grade distribution:

- 20% — distinction (3.5-4.0): one strong, one Stretch-completing
- 60% — pass (3.0-3.4): the modal "did the work" group
- 15% — pass with feedback (2.5-2.9): rushed but complete
- 5%  — borderline or below: needs a conversation

If you're seeing mostly 4.0s or mostly 2.0s in your first batch,
recalibrate against `MP2_Sample_Submission.md`.

---

*"You see, but you do not observe."* — Sherlock Holmes
