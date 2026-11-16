# Launch advisor: the web service (xviral/app). CPU only.
#
#   docker build -f virality/Dockerfile -t launch-advisor .        # from the repository root
#   docker run -p 8000:8000 -v /path/to/data:/data \
#     -e XVIRAL_CHECKPOINTS=/data/runs/ts46_s1/best.pt,/data/runs/ts46_s2/best.pt,/data/work/model/compare_v2/full/best.pt \
#     -e XVIRAL_POSTS=/data/work/hygiene/posts.parquet -e XVIRAL_SID=/data/work/emb/sid_codebook.npz launch-advisor
#
# Models and the corpus are mounted, not baked in. Every setting is an environment variable (docs/launch_advisor_api.md).
ARG BASE=python:3.12-slim
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# xviral reads X's own code by path (post rendering in phoenix/reference, ranking weights in home-mixer and
# vm-ranker), so the repository layout is kept.
WORKDIR /repo/virality
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY phoenix/reference /repo/phoenix/reference
COPY home-mixer/params/param.rs /repo/home-mixer/params/param.rs
COPY vm-ranker/params.rs /repo/vm-ranker/params.rs
COPY virality/pyproject.toml ./
COPY virality/xviral ./xviral
COPY virality/research/15_pair_confidence.json ./research/15_pair_confidence.json
RUN pip install -e ".[app]" "sentence-transformers>=5.1" "transformers>=4.57" \
 && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN useradd --create-home --uid 10001 app && chown -R app /opt/hf /repo
USER app
ENV XVIRAL_SCALES=/repo/virality/research/15_pair_confidence.json XVIRAL_PRELOAD=1 HF_HUB_OFFLINE=1 XVIRAL_RATE_PER_MIN=30
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/ready').status==200 else 1)"
CMD ["uvicorn", "xviral.app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
