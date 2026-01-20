# Bilingual Merge (df-diff)

A tool for intelligently merging bilingual datasets by detecting and filtering unique entries using exact matching, fuzzy matching, and semantic similarity.

## Overview

This tool compares two bilingual datasets (with English and French columns) and merges unique rows from the source into the target dataset. It uses a three-stage filtering process:

1. **Exact matching**: Identifies rows that don't exist in the target
2. **Fuzzy matching**: Filters out rows with high string similarity using RapidFuzz
3. **Semantic matching**: Uses embeddings to detect semantically similar entries

The final output is a deduplicated JSONL file containing the merged dataset.

## Installation

### Base Installation

```bash
pip install -e .
```

Then set your API key:

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

## Usage

### Basic Example

```bash
bilingual-merge \
  --source source.parquet \
  --target target.parquet \
  --en-col en \
  --fr-col fr \
  --out merged.jsonl \
  --fuzzy-threshold 92 \
  --semantic-threshold 0.82 \
  --embed-backend minilm
```

### Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Source dataset path (parquet/csv/jsonl/json) | *required* |
| `--target` | Target dataset path (parquet/csv/jsonl/json) | *required* |
| `--out` | Output JSONL file path | *required* |
| `--en-col` | Name of the English column | `en` |
| `--fr-col` | Name of the French column | `fr` |
| `--fuzzy-threshold` | Keep rows whose best fuzzy match score is below this (0-100) | `92` |
| `--semantic-threshold` | Keep rows whose best cosine similarity is below this (0-1) | `0.82` |
| `--embed-backend` | Embedding backend: `minilm` or `gemini` | `minilm` |
| `--minilm-model` | MiniLM model identifier | `sentence-transformers/all-MiniLM-L6-v2` |
| `--gemini-model` | Gemini embedding model identifier | `text-embedding-004` |
| `--gemini-api-key` | Gemini API key (optional if `GEMINI_API_KEY` env var is set) | `None` |
| `--max-candidates-per-row` | Maximum target rows to scan per candidate (speed cap) | `200` |

### Supported File Formats

The tool supports the following input formats:
- Parquet (`.parquet`)
- CSV (`.csv`)
- JSONL (`.jsonl`)
- JSON (`.json`)

Output is always JSONL format.

## How It Works

1. **Read & Normalize**: Both datasets are loaded and normalized
2. **Exact Diff**: Rows present in source but not in target are identified
3. **Fuzzy Filter**: Candidates are filtered using string similarity (RapidFuzz)
4. **Semantic Filter**: Remaining candidates are filtered using embeddings
5. **Merge & Dedupe**: Unique rows are appended to target and deduplicated
6. **Output**: Final merged dataset is written as JSONL

The tool provides progress summaries at each stage showing how many rows remain after each filtering step.

## Examples

### Using MiniLM (Local Embeddings)

```bash
bilingual-merge \
  --source data/source.csv \
  --target data/target.csv \
  --out results/merged.jsonl \
  --embed-backend minilm \
  --fuzzy-threshold 90 \
  --semantic-threshold 0.85
```

### Using Gemini (API-based Embeddings)

```bash
bilingual-merge \
  --source data/source.parquet \
  --target data/target.parquet \
  --out results/merged.jsonl \
  --embed-backend gemini \
  --gemini-api-key "your-key-here" \
  --fuzzy-threshold 95 \
  --semantic-threshold 0.80
```

## Requirements

- Python >= 3.13
- See `pyproject.toml` for full dependency list