# Autonomous Data Science Research Agent

This project demonstrates how to orchestrate the OpenAI Agent API to build an
end-to-end research assistant tailored for data science teams. The pipeline
accepts a user query, automatically investigates relevant websites, stores the
findings inside a local vector database, and produces a concise executive
summary.

## Features

- **Autonomous web research** powered by an OpenAI agent with the web search
  tool enabled.
- **Embedding & retrieval pipeline** that indexes research snippets in a local
  vector store for future reuse.
- **Summarisation layer** that distils the aggregated findings into a polished
  report tailored to data science stakeholders.
- **CLI workflow** for triggering research sprints from the terminal.

## Project structure

```
.
├── datasci_tool/
│   ├── config.py          # Configuration dataclasses
│   ├── embeddings.py      # Embedding generation helpers
│   ├── pipeline.py        # High level research workflow
│   ├── research_agent.py  # OpenAI Agent API wrapper
│   ├── summary.py         # Summary generation helper
│   └── vector_store.py    # Lightweight vector database
├── scripts/
│   └── run_pipeline.py    # CLI entrypoint
├── tests/
│   └── test_pipeline.py   # Unit tests with heavy mocking
└── pyproject.toml
```

## Getting started

1. Install dependencies (preferably in a virtual environment):

   ```bash
   pip install -e .[dev]
   ```

2. Export your OpenAI credentials:

   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. Run a research sprint from the command line:

   ```bash
   python scripts/run_pipeline.py "Bayesian optimization for hyperparameter tuning"
   ```

   The command prints a JSON payload containing the agent's summary, the sources
   it discovered, and the top similar items retrieved from the vector database.

## Testing

The test-suite uses mocks to avoid real API calls. Execute it with:

```bash
pytest
```

## Notes on the OpenAI Agent API

- The agent is instantiated with the `web_search` and `code_interpreter` tools
  so it can browse the web and run lightweight calculations as needed.
- Research results are streamed back as a JSON array to ensure downstream
  components receive structured information.
- Embeddings are generated with `text-embedding-3-large` and stored inside a
  lightweight JSON-backed vector store for simplicity. You can swap in a
  managed vector database (such as Pinecone or Chroma) by implementing the same
  interface as `LocalVectorStore`.
