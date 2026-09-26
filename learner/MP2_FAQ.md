# MP2 · FAQ

> *Questions we expect you to hit. Skim this before posting in the cohort chat.*

---

## Setup

**Q: I get `OPENAI_API_KEY not set` when I run `python mp2_rag.py ingest`.**

Did you `source .env`? In a new terminal window, you need to run that again
— `source` only affects the current shell session.

Check: `echo $OPENAI_API_KEY` should print your key. If it's empty, your
`.env` file is missing the `export` keyword. Open `.env` and confirm each
line starts with `export`.

**Q: `qdrant_client.http.exceptions.UnexpectedResponse: Service unavailable`**

The Qdrant free-tier cluster goes to sleep after ~30 minutes of inactivity.
The first request after sleep takes ~60 seconds. Wait, retry. If it still
fails after 2 minutes, your cluster URL may be wrong — log in to
[cloud.qdrant.io](https://cloud.qdrant.io) and check.

**Q: Can I use a local Qdrant instead?**

Yes. Run `docker run -p 6333:6333 qdrant/qdrant` and set
`QDRANT_URL=http://localhost:6333` in your `.env`. Comment out
`QDRANT_API_KEY` (the local server doesn't require one). Same code,
no other changes.

**Q: I'm on Windows / `source .env` doesn't work.**

Use `dotenv` instead — `pip install python-dotenv`, then at the top of
`mp2_rag.py` add:

```python
from dotenv import load_dotenv
load_dotenv()
```

Make sure your `.env` does NOT have `export` if you're using `dotenv`
(it only parses `KEY=value` lines, not shell commands). Pick one approach
or the other, not both.

---

## Chunking

**Q: How big should my chunks be?**

The reference solution uses 500 characters with 80-char overlap. These are
reasonable defaults for narrative prose. Smaller = more precise retrieval but
each chunk has less context. Bigger = more context per chunk but worse
recall on specific facts. If your answers are missing details, try smaller
chunks. If your answers are missing surrounding context, try bigger.

**Q: Do I need to split on paragraphs, or are fixed-size windows OK?**

Fixed-size windows work fine for MP2. The reference solution uses paragraph-
aware chunking because it's smarter and the stories have clear paragraph
breaks, but a simple `text[i:i+500]` loop is acceptable. The rubric doesn't
require sophisticated chunking; it requires *working* chunking.

**Q: How many chunks should I end up with?**

The reference solution produces ~70-80 chunks across the 5 stories. If
you're getting fewer than ~30 or more than ~200, your chunking has a bug.
Print the chunks and look.

---

## Embeddings

**Q: Can I use a free embedding model instead of OpenAI?**

Not for this project. The Brief mandates OpenAI to match the capstone
constraint. The local alternatives (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
do work, but we want consistency across the cohort. After MP2 ships, feel
free to try a local model in your own time and add it to the reflection.

**Q: Should I cache embeddings?**

The corpus is small enough that you don't need to. The whole corpus
embeds in one API call (one round-trip), costs <$0.01, and the script
only ingests once. The capstone's embedding cache (from W7) was for a
bigger corpus you re-ingest often.

**Q: My embedding call is failing with "max tokens exceeded".**

OpenAI's embedding API accepts batches of up to ~2048 inputs but
each input is capped at ~8K tokens. If a single chunk exceeds 8K tokens,
the batch fails. With 500-char chunks, you won't hit this. If you tried
to embed each whole story as one chunk, you might. Reduce chunk size.

---

## Retrieval

**Q: My retrieval keeps pulling the wrong story.**

Inspect the chunks it's pulling. The most common cause: a chunk in story
A contains text that's semantically similar to your question about story B.
Example: a question about "the snake" pulls a chunk from the Engineer's
Thumb that mentions "the vine he climbed down" because both involve climbing
plant-like things. Fix: be more specific in your question, or use longer
chunks that carry more context.

**Q: Should I use k=3 or k=5 or k=10?**

k=3 is fine for MP2. The reference solution uses k=3. Bigger k gives the
LLM more context but also more distractor chunks. If your answers are
generic, try smaller k. If your answers are missing a key fact, try bigger.

**Q: Do I need metadata filtering?**

No. MP2 doesn't require it. The capstone's filter DSL (W8) was for
heterogeneous corpora where some docs were HR, some IT, some legal.
A pure-Sherlock corpus has no such partition.

---

## Generation

**Q: My LLM is hallucinating — it's making up details not in the corpus.**

Three causes, in order of likelihood:

1. **Your system prompt is too permissive.** "Answer the question" lets the
   LLM fall back on its training knowledge of Sherlock Holmes (it knows
   the originals!). Use the system prompt from the starter, which says
   "use ONLY the provided excerpts."
2. **Your retrieval missed the answer chunk.** With no relevant context
   provided, the LLM falls back on training data. Print the retrieved
   chunks and check.
3. **The model is gpt-4o-mini and the question is hard.** Switch to gpt-4o
   (more expensive but smarter). Note this in your reflection.

**Q: Should I use temperature=0?**

Yes. The reference solution uses 0.0. RAG answers should be deterministic
given the same context. Higher temperatures introduce variance that makes
debugging harder.

**Q: How do I get citations?**

Include the source in the context you pass to the LLM, then ask the system
prompt to cite. The reference solution prepends each retrieved chunk with
`[Source: <title> — <section>]` — the LLM picks this up and includes it
in the answer.

---

## Validation

**Q: My validate output shows ✗ for q1 — what does that mean?**

The retrieved chunks didn't include the expected source story. Either:

- Your retrieval missed the right chunk (most common)
- The expected source in the JSONL is wrong (rare — check)
- Your chunking is misclassifying the source (look at `chunk["source"]`)

Print the retrieved chunks for the failing question. The actual content
will tell you what's wrong.

**Q: My validate shows ✓ on source but the "facts matched" count is low.**

The expected_facts check is a generous substring match — it just looks
for the lowercase fact in the answer text. Low fact-match can mean:

- The answer is correct but phrased differently (e.g., "venomous snake" instead
  of "swamp adder"). This is fine — the rubric doesn't grade on facts_matched.
- The answer is missing important details. Adjust your system prompt or
  use larger k.

`facts_matched` is a *diagnostic*, not a grade. ✓ on source is what matters.

---

## The Stretch

**Q: Should I do the Stretch (W9 hybrid+rerank)?**

Only if you've finished everything else with time to spare. The Stretch is
genuinely optional and won't increase your grade. Its value is portfolio
storytelling: "I built a basic RAG and then I extended it with hybrid
retrieval and a cross-encoder reranker, here's the precision delta."

**Q: Does hybrid+rerank help on the Sherlock corpus?**

Try it and see — that's the point. Hybrid retrieval shines on corpora with
rare exact-match terms (product codes, version numbers, technical IDs).
Narrative prose is less obviously a fit. Your reflection on this could be
genuinely interesting.

---

## Submission

**Q: Can I submit a Jupyter notebook instead of a script?**

The brief says script. If you have a strong reason to use a notebook (you've
been working in notebooks all along, your reflection benefits from inline
output), submit both — `mp2_rag.py` AND `mp2_rag.ipynb`. The script is what
gets graded for code-cleanliness.

**Q: I used my own corpus. How do I submit?**

In your reflection, name the corpus and say why you chose it. Include the
data in your repo (if it's small + freely shareable) or link to the source.
Your 5 Q&A pairs should be against your corpus, not the Sherlock one.

**Q: I didn't finish all 5 questions. Can I still submit?**

Submit what you have with a note in the reflection. Partial credit beats no
submission. The rubric is generous on completion: 4 working questions + an
honest reflection is a passing submission.

---

## Beyond MP2

**Q: What's next?**

W10 — caching + KB lifecycle. The plumbing on top of W9's stack. Normal
difficulty after the DANGER ZONE of W9.

**Q: How does MP2 relate to M2 (the Phase 2 milestone at W12)?**

M2 is the production-ready capstone RAG with eval framework. MP2 is the
warm-up. By M2 you'll be doing what MP2 did, but on the capstone corpus,
with hybrid+rerank as default, with a real eval harness, and defending it
at a design review. MP2 builds the muscle; M2 stress-tests it.
