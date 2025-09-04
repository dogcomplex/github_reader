# GitHub Reader

GitHub Reader is a small toolkit that fetches recent GitHub repository READMEs (by activity) and builds a hierarchical, batched summarization pipeline over them using a local OpenAI-compatible HTTP API for chat completions and embeddings.

## Overview

- Fetch recent repositories via BigQuery (`githubarchive.month.*`) and download their README text to `readmes/<yyyymm>/`.
- Process READMEs in iterative levels (`summaries/level_<N>/`) to produce:
  - Chunked content for large files
  - Per-document summaries, tags, and reports
  - Optional embeddings and grouping with K-Means
  - Optional group-level summaries

The pipeline favors practical throughput, simple file-based caching, and a consistent filename scheme based on a 6-digit priority prefix and a normalized name.

## Repo Layout

- `last_months_readmes.py`: Fetches recent repositories (BigQuery) and downloads their README content via the GitHub API.
- `hierarchical_summary.py`: Entry point for the multi-level metadata pipeline using async I/O.
- `metadata_config.py`: Declares `DEFAULT_METADATA_CONFIGS` for each metadata type (CHUNK, SUMMARY, TAGS, REPORT, EMBEDDING, GROUP, GROUP_SUMMARY, LANGUAGE, TRANSLATION, SPLIT_SUMMARY).
- `metadata_processor.py`: Orchestrates metadata processing according to dependency order; returns in-memory results. (Chunking writes to disk.)
- `metadata_processors.py`: Concrete async processors for each metadata type; includes LLM and embedding calls.
- `models.py`: Data classes and path helper for output naming.
- `group_summaries.py`: Legacy/alternate grouping and embedding logic.
- `summarize.py`: Legacy summarization helper (single-document/grouper utilities).
- `processing_context.py`: Small context struct used during processing.
- `alt_scraper.py`: A WIP example (Node.js-style code) misnamed with a `.py` extension; not used by the pipeline.
- `readmes/`, `summaries/`, `old/`: Generated inputs/outputs, caches, and prior experiments.

## Requirements

- Python 3.10+
- A local OpenAI-compatible API endpoint for chat and embeddings at `http://localhost:1234/` with:
  - Chat model alias: `local-model` (OpenAI Chat Completions format)
  - Embedding model: `text-embedding-bge-m3`
- Google BigQuery access to `githubarchive.month.*` (for README discovery)

### Suggested Python packages

Install via pip (example):

- `aiohttp`
- `requests`
- `tqdm`
- `numpy`
- `scikit-learn`
- `tenacity`
- `beautifulsoup4`
- `langdetect`
- `python-dotenv`
- `google-cloud-bigquery`
- `google-auth`

## Environment Variables

Place these in `.env` or export them in your environment:

- `GITHUB_TOKEN`: GitHub API token (for README fetches)
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Absolute path to a GCP service account JSON

## How to Run

1) Fetch READMEs for the current month:

- Ensure you have `GOOGLE_APPLICATION_CREDENTIALS` set and access to `githubarchive.month.<YYYYMM>`.
- Run: `python last_months_readmes.py`
  - Outputs to `readmes/<yyyymm>/` (e.g., `readmes/202411/`).

2) Run the hierarchical summarization pipeline:

- Ensure your local LLM server is running at `http://localhost:1234` for both chat and embeddings.
- Run: `python hierarchical_summary.py`
  - Outputs go under `summaries/level_<N>/` in subdirectories such as `chunks`, `summaries`, `tags`, `reports`, `embeddings`, `groups`, `language`, `translations`.

## Outputs and Naming

- Files use a 6-digit left-padded priority prefix derived from activity (e.g., `001234_repo_owner_repo_name_*`).
- Level > 1 files include `_<level>` before the metadata suffix.
- Example: `000123_user_repo_2_SUMMARY.md`.

## What Likely Works

- README fetching (`last_months_readmes.py`) with valid GCP credentials and a GitHub token.
- Single-document metadata (SUMMARY, TAGS, REPORT, LANGUAGE, TRANSLATION, CHUNK) through `metadata_processor.py` → `metadata_processors.py` using the local LLM/embedding endpoints.
- Multi-level iteration in `hierarchical_summary.py` that falls back to per-document summaries when grouping is not available.

## What Needs Attention / Caveats

- Grouping and Embeddings on Disk:
  - `process_groups` expects embedding `.npy` files on disk under `summaries/level_<N>/embeddings/`, but the standard flow currently returns embeddings in-memory and does not persist them by default. As a result, grouping will often be skipped and the pipeline will fall back to per-document summaries. To enable grouping, add persistence for `EMBEDDING` outputs.
- File-Naming Consistency:
  - Grouping code builds embedding paths using only `doc.name`; while the configured filenames include the priority and sometimes level. Aligning these conventions is recommended before relying on grouping.
- `alt_scraper.py` contains Node.js-style code but uses a `.py` extension; it is a WIP/placeholder and not runnable as Python.
- Legacy modules (`summarize.py`, `group_summaries.py`, `hierarchical_summary_old.py`) are retained for reference and may not be fully integrated with the current pipeline.

## Roadmap Ideas

- Persist all metadata outputs (including embeddings, summaries, tags) to disk consistently, then wire grouping to consume those artifacts reliably.
- Unify filename conventions for all processors and levels.
- Add a `requirements.txt` and optional `Makefile`/invoke tasks for repeatable runs.
- Add unit tests for filename parsing, chunking, and grouping behaviors.

## License and Third-Party Content

- Downloaded READMEs under `readmes/` are third-party content subject to their respective repository licenses. Do not redistribute without confirming license terms.
- Generated summaries/tags/reports may be derivative of the source READMEs; assess licensing for distribution.

## Security Notes

- Do not commit API keys or cloud credentials. Keep `.env` and any service account files out of git.
- If a service account JSON has been committed, revoke/rotate it immediately and remove it from git history before publishing.
