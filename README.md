> [!CAUTION]
> This repository is under development.

## pydev-ai

Production-ready base project for generating and evolving Python codebases with CrewAI Flows. Includes flows to create a project from scratch and to iterate on an existing one, with cost guardrails, persistent RAG, and QA tools.

### Features
- **Two Flows** (CrewAI Flows): `new` (greenfield) and `iterate` (changes on existing repo)
- **Persistent knowledge**: summaries per file/module + local RAG (ChromaDB)
- **Agent architecture**: design, development (junior/senior/lead), fix integrator, summary generator
- **Flow guardrails**: token limit per response and bounded debugging loops
- **Quality tools**: Black, Ruff, pytest, test framework detection
- **Dockerized** with `Makefile` for simplified commands

---

## Requirements
- Docker and Docker Compose (recommended), or Python 3.11+ local
- Environment variable `OPENAI_API_KEY` configured (OpenAI)

Optional: `.env` file in the root. This project loads variables with `python-dotenv`.

---

## Quick start (Docker Compose)

```bash
# Clone the repo
git clone <repository-url>
cd crewai-python-dev

# Export your API key
export OPENAI_API_KEY="your_api_key"

# Build the image
docker compose build app

# Create a new project
docker compose run --rm app new \
  --prompt "REST API for library with inventory and search" \
  --out /workspace/out/bookstore

# Iterate on an existing project
docker compose run --rm app iterate \
  --prompt "Add CSV import endpoint" \
  --repo /workspace/out/bookstore

# Format, lint and tests
docker compose run --rm app fmt  --repo /workspace/out/bookstore
docker compose run --rm app lint --repo /workspace/out/bookstore
docker compose run --rm app test --repo /workspace/out/bookstore
```

Notes:
- The `./outputs` volume from the host is mounted as `/workspace` inside the container. If you use `--out /workspace/out/<name>`, you'll see the results in `./outputs/out/<name>`.

---

## Local execution (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# General help
python -m src.app --help

# Create new project
python -m src.app new \
  --prompt "Create a REST API for a library" \
  --out ./outputs/bookstore

# Iterate on an existing project
python -m src.app iterate \
  --prompt "Add bulk CSV import" \
  --repo ./outputs/bookstore

# Format, lint and tests
python -m src.app fmt  --repo ./outputs/bookstore
python -m src.app lint --repo ./outputs/bookstore
python -m src.app test --repo ./outputs/bookstore
```

---

## Environment variables

By default they are read from your environment or from `.env` (thanks to `python-dotenv`).

| Variable | Description | Default (Local) | Default (Docker Compose) |
|---------|-------------|------------------|--------------------------|
| `OPENAI_API_KEY` | OpenAI API key | "" | "" (you must set it) |
| `LOG_LEVEL` | Logging level | `INFO` | `INFO` |
| `OPENAI_MODEL_LIGHT` | Light model | `gpt-5-nano` | `gpt-4o-mini` |
| `OPENAI_MODEL_REASONING` | Reasoning model | `gpt-5-nano` | `gpt-4o` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` | `text-embedding-3-small` |
| `PYTEST_TIMEOUT` | pytest timeout (s) | `1800` | `1800` |

Quick example:
```bash
export OPENAI_API_KEY="your_api_key"
export LOG_LEVEL="INFO"
# Others can stay with default or be overridden
```

---

## Available CLI

```bash
python -m src.app --help
```

Commands:
- `new`     Creates a project from scratch from a short prompt
- `iterate` Applies changes to an existing repo
- `fmt`     Formats code (Black)
- `lint`    Linting (Ruff)
- `test`    Runs tests (pytest)

All accept `--repo` or `--out` options as appropriate.

---

## Makefile (shortcuts)

```bash
# Expected variables
export OPENAI_API_KEY="your_api_key"
export PROMPT="Create a REST API for a library"
export NAME="bookstore"

# Targets
make build
make new
make iterate
make fmt
make lint
make test
```

Note: there is an `index` target in the `Makefile`, but the flow already indexes automatically when appropriate (RAG). You don't need to run it manually.

---

## Project structure

```
crewai-python-dev/
├── docker-compose.yml
├── docker/
│   └── Dockerfile
├── requirements.txt
├── Makefile
├── README.md
└── src/
    ├── app.py                # CLI (typer)
    ├── settings.py           # Configuration (env, models, paths)
    ├── flows/
    │   ├── new_project_flow.py
    │   ├── iterate_flow.py
    │   └── utils.py          # Utilities: parsing, writing, digests
    ├── crews/
    │   ├── design/           # Project design
    │   ├── development/      # Development by levels (junior/senior/lead)
    │   ├── fix_integrator/   # Integrates suggested fixes
    │   ├── summaries/        # Generates summaries per file/module
    │   └── iterate/          # Crew for iteration
    ├── tools/
    │   ├── rag_tools.py      # Indexing and semantic search (Chroma)
    │   └── test_runner.py    # pytest/unittest wrappers and detection
    ├── summaries/
    │   ├── summarizer.py
    │   └── storage.py
    └── utils/
        └── routing.py        # LLM selection
```

---

## Flows (CrewAI Flows)

### `NewProjectFlow`
1. Prepares the output directory
2. Executes `ProjectDesignCrew` (high-level design by files)
3. For each design block, executes the appropriate development crew (junior/senior/lead)
4. Integrates fixes (if any) and generates summaries per file/module
5. Writes code in `<out>/src` and summaries in `<out>/summaries`
6. Formats and lints (Black + Ruff)

Note: there is an additional QA/tests/docs phase in `BuildCrewPhase2` ready to be used by stages. In the current flow, the test/doc writing part is prepared but disabled; you can activate it according to your needs.

### `IterateFlow`
1. Initializes knowledge (digests + RAG) if missing
2. Executes `IterateCrew` with token limits and bounded debugging loops

---

## Knowledge and RAG
- Persistent local vector store: ChromaDB in `data/knowledge/vectors/`
- Tool-controlled indexing (`rag_index_repo`) with globs for `*.py`, `*.md`, `*.txt`
- Semantic search (`rag_search`) to provide compact context to agents
- File/module summaries are saved in `<out>/summaries` and facilitate future iterations

---

## Output structure

After `new` or `iterate`, the output directory contains at minimum:
- `<out>/src/` generated/updated source code
- `<out>/summaries/` summaries per file and module

If you run inside Docker with `--out /workspace/out/<name>`, on the host you'll find `./outputs/out/<name>`.

---

## Troubleshooting
- Verify `OPENAI_API_KEY`
- In Docker, adjust resources if working with large projects
- Enable verbose logs with `LOG_LEVEL=DEBUG`
- If a repo has no tests, `test` will still run `pytest -q` (may not find anything)

```bash
export LOG_LEVEL=DEBUG
```

---

## License
Define the license you prefer (MIT/Apache-2.0, etc.).
