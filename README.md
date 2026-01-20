### Install

Base:
pip install -e .

MiniLM embeddings:
pip install -e ".[minilm]"

Gemini embeddings:
pip install -e ".[gemini]"
export GEMINI_API_KEY="..."

### Run

bilingual-merge \
  --source a.parquet \
  --target b.parquet \
  --en-col en \
  --fr-col fr \
  --out merged.jsonl \
  --fuzzy-threshold 92 \
  --semantic-threshold 0.82 \
  --embed-backend minilm
