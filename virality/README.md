# xviral: the data layer for the virality engine

Design: [DESIGN.md](DESIGN.md) (evidence, budget, gates), [ARCHITECTURE.md](ARCHITECTURE.md)
(stages S0 to S7), [DATA_PIPELINE.md](DATA_PIPELINE.md) (sources, filters, labels, every
feature and why). Research behind them: [research/](research/).

This package implements the data pipeline: ingesting every source into one schema, filtering
to the target audience, deduplicating, building labels and features that follow the X
algorithm in this repository, and the GPU stages that compute X's own post embedding and
semantic IDs. Everything ported from X cites the source file in `xviral/constants.py` or in the
module docstring. Code from `phoenix/reference/` (the post encoder and the codebook trainer) is
imported from the repository, not copied.

## Install

```bash
cd virality
python -m venv .venv
```

```bash
.venv/bin/pip install -e ".[dev]"
```

On the GPU machine, add the encoders and the codebook trainer:

```bash
.venv/bin/pip install -e ".[encoders,codebook,ann]"
```

## Get the data

USC 2024 corpus (research use only, CC BY-NC-SA). The repository is about 20 GB; a blobless
clone fetches files on demand:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/sinking8/usc-x-24-us-election ../../sinking8/usc-x-24-us-election
```

Community Archive tables over its public REST API. The anon key is published in the archive's
own `docs/api-doc.md`; export it, then download:

```bash
export CA_ANON_KEY='<key from community-archive docs/api-doc.md>'
```

```bash
.venv/bin/xviral ca-download --config configs/pipeline.yaml
```

The 2026 view corpus and the 2023 timeline corpus come from Hugging Face; `scripts/gpu_run.sh`
downloads both and `configs/gpu.yaml` points at them. On a machine without the disk for the view
corpus, `xviral.sources.nitter_sample.ingest_streaming` filters it one shard at a time.

The tech scrape: put the JSONL files where `paths.user_scrape` points. Analytics exports: list
the CSV files under `paths.analytics`.

Accounts are kept or dropped whole by the `select` block of the config (minimum posts, strict
tech share, cap per account, optional general accounts): see DATA_PIPELINE.md section 15.3.

## Run

Phase 0 runs on CPU and ends with the go/no-go numbers:

```bash
.venv/bin/xviral phase0 --config configs/pipeline.yaml
```

Or stage by stage: `ingest`, `unify`, `filter`, `authors`, `hygiene`, `dedup`, `relevance`,
`features`, `dataset`, `report`, `baseline`. Outputs land under `paths.work/<stage>/`.
After `relevance`, label `relevance/audit_to_label.csv` (fill the `relevant` column, save as
`audit_labeled.csv`) and rerun `relevance`: the threshold is then set for the target precision.

Phase 1 runs on the GPU box, in this order: `media-select`, `fetch-images`, `fetch-videos`,
`asr`, `embed-mm`, `sid-train`, `sid-assign`, `embed-siglip`, `neighbors`.

Without a large GPU, `embed-text` replaces `embed-mm`: a small text encoder (`embed.text_model`)
writes to the same store and every later stage runs unchanged. That is the development path, not
the production one.

New candidate features are vetted before they become model inputs:

```bash
.venv/bin/python scripts/vet_features.py --table data/work/model/table --rows data/work/dataset/rows.parquet --out vet.json
```

## Model

Design: [MODEL.md](MODEL.md). After the embedding stages:

```bash
.venv/bin/xviral model-prepare --config configs/pipeline.yaml
```

```bash
.venv/bin/xviral model-train --config configs/pipeline.yaml --preset base --device cuda
```

```bash
.venv/bin/xviral model-eval --config configs/pipeline.yaml --device cuda
```

To compare the model with its ablations and the baselines on the time-forward test split:

```bash
.venv/bin/python scripts/compare_models.py --table data/work/model/table --out data/work/model/compare --preset base --device cuda
```

To score drafts for an account with a trained model:

```python
from xviral.serve import DraftScorer

scorer = DraftScorer("data/work/model/runs/base/best.pt", embed_fn)   # embed_fn: list of texts -> array, the training encoder
reports = scorer.score(history, [{"text": "draft one"}, {"text": "draft two"}], {"handle": "acme", "followers": 12_000})
```

`history` is the account's past posts in the unified schema (any source reader in `xviral/sources` produces it).

## Test

```bash
.venv/bin/python -m pytest -q
```

The suite includes the X repository's own test cases for weighted text length, a check that
batched encoder inputs are identical to what `phoenix/reference/mm_encoder.py` sends to the
model, leakage tests for every trailing feature and for neighbour retrieval, a brute-force check
of pair construction, real ffmpeg video sampling, and a full phase-0 run across all four source
formats.
