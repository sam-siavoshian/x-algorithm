# 07 — Compute, data and budget plan for a multimodal X-post engagement/virality model

Research date: 2026-11-07. Prices are us-east-1 list/spot snapshots from third-party trackers and
vendor pricing pages collected today; spot and marketplace prices move daily, so re-check the day you
launch. Throughput numbers marked **est.** are first-principles estimates (FLOPs / sustained TFLOPS at a
stated utilisation) anchored to the few published measurements cited; they are planning numbers, and
the first task on any GPU should be a 200-step benchmark to replace them.

Hardware options on the table: (a) RTX 2060 6 GB; (b) Mac mini, Apple Silicon M4 / M4 Pro, 16-64 GB
unified memory (note: Apple replaced the M4 mini with M6 / M5 Pro minis on 2026-08-25; see §2.5);
(c) a few hundred dollars of AWS credits.

---

## 1. GPU prices (late 2026)

### 1.1 AWS EC2, us-east-1, Linux, per instance-hour

| Instance | GPU (VRAM) | vCPU / RAM | On-demand | Spot (lowest AZ, Nov 2026) | Spot per GPU |
|---|---|---|---|---|---|
| g4dn.xlarge | 1x T4 (16 GB) | 4 / 16 GB | $0.526 | ~$0.23 (range $0.16-0.26) | $0.23 |
| g5.xlarge | 1x A10G (24 GB) | 4 / 16 GB | $1.006 | ~$0.47-0.50 | $0.49 |
| g5.2xlarge | 1x A10G (24 GB) | 8 / 32 GB | $1.212 | ~$0.55-0.67 | $0.55 |
| g6.xlarge | 1x L4 (24 GB) | 4 / 16 GB | $0.805 | ~$0.44-0.59 | $0.44 |
| g6e.xlarge | 1x L40S (48 GB) | 4 / 32 GB | $1.861 | ~$1.32 | $1.32 |
| p4d.24xlarge | 8x A100 **40 GB** (p4de = 80 GB) | 96 / 1152 GB | $21.96 | ~$8.91-14.32 | $1.11-1.79 |
| p5.48xlarge | 8x H100 80 GB | 192 / 2048 GB | $55.04 ($6.88/GPU) | ~$20.8-30.6 | $2.6-3.8 |
| p5.4xlarge | 1x H100 80 GB (newer single-GPU size) | 16 / 256 GB | ~$6.88 (unverified; 1/8 of p5.48xlarge) | n/a | — |

Notes
- AWS cut P5 on-demand 44% and P4d 33% in June 2025; these are the post-cut rates. Some trackers
  report Capacity Block prices rising ~15% (Jan 2026) and ~20% (Jul 2026), which does not affect
  on-demand/spot.
- p4d is 40 GB A100s; for 80 GB on AWS you need p4de or p5. For a QLoRA 7B VLM 40 GB is plenty.
- GPU spot interruption rates in us-east-1 are materially higher than CPU spot (trackers quote
  5-10%/hour for p4d in busy periods; ~10%/month average across us-east-1 vs <1% in us-west-1).
  Checkpoint every 10-15 min and use Spot Instance Advisor / placement scores before launching;
  us-west-2 / us-east-2 often have better GPU spot availability.
- Sources: [DoiT g6e.xlarge](https://www.doit.com/compute/compute/aws/us-east-1/g6e.xlarge),
  [Holori g6e.xlarge](https://calculator.holori.com/aws/ec2/g6e.xlarge),
  [DevZero g6.xlarge](https://www.devzero.io/instances/aws/g6.xlarge),
  [Vantage g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge),
  [DoiT g5.xlarge](https://www.doit.com/compute/compute/aws/us-east-1/g5.xlarge),
  [DoiT g5.2xlarge](https://www.doit.com/compute/spot/us-east-1/g5.2xlarge),
  [DoiT g4dn.xlarge](https://www.doit.com/compute/spot/us-east-1/g4dn.xlarge),
  [DoiT p4d.24xlarge](https://compute.doit.com/spot/us-east-1/p4d.24xlarge),
  [DoiT p5.48xlarge](https://www.doit.com/compute/spot/us-east-1/p5.48xlarge),
  [CloudOptimo on the 2025 price cuts](https://www.cloudoptimo.com/blog/aws-announces-significant-price-reductions-for-nvidia-gpu-ec2-instances/),
  [Thunder Compute EC2 GPU guide Oct 2026](https://www.thundercompute.com/blog/ec2-gpu-instances),
  [Spot interruption discussion](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability).

### 1.2 Marketplace / neocloud alternatives (per GPU-hour)

| Provider | A100 80 GB | H100 80 GB | Others | Date / note |
|---|---|---|---|---|
| Vast.ai (cheapest verified-ish listings) | $1.00 PCIe / $1.07 SXM (40 GB: $0.60-0.67) | $2.20 SXM / $2.60 PCIe | L40S $0.60, L4 $0.46, RTX 4090 $0.38, RTX 3090 $0.19, A6000 $0.46 | 2026-11-06, [computeprices.com/providers/vast](https://computeprices.com/providers/vast); medians ~$0.80-1.00 (A100) and ~$2.20 (H100) per [JouleMath Oct 2026](https://joulemath.com/blog/gpu-rental-prices-h100-h200-a100-rtx-4090-october-2026/) |
| RunPod Community Cloud | from $1.19 PCIe / $1.39 SXM | from $1.99 PCIe / $2.69 SXM | L40S $0.79, L4 $0.44, 4090 $0.34, A6000 $0.33 | [runpod.io/pricing](https://www.runpod.io/pricing), fetched today |
| RunPod Secure Cloud | $1.79 | $2.89 PCIe / $3.99 SXM | L40S $1.09, L4 $0.59, 4090 $0.89 | same |
| Lambda (self-serve instances) | not offered at 1x; 8x HGX A100 80 GB $3.19/GPU; A100 40 GB 1x $1.99 | 1x H100 SXM $4.89, 1x H100 PCIe $3.79 | A10 24 GB $1.29, GH200 96 GB $2.59, B200 $7.99 | New price list effective 2026-11-16 per [lambda.com/pricing](https://lambda.com/pricing); earlier 2026 list was H100 SXM $3.99 (8x) / $4.29 (1x), H100 PCIe $3.29, A100 80 GB $2.79 ([Thunder Compute](https://www.thundercompute.com/blog/lambda-labs-vs-thunder-compute)) |

Takeaways: for a <$500 budget Lambda is not competitive; Vast.ai (interruptible or verified
datacenter hosts) and RunPod Community are 2.5-6x cheaper per A100/H100 hour than AWS on-demand and
comparable to AWS *spot*, with the advantage that you can rent a single 80 GB A100/H100 instead of an
8-GPU p4d/p5 node. The sponsor's AWS credits only work on AWS, so the realistic plan is "AWS credits for
the small-GPU phases, ~$100-150 of cash on Vast/RunPod for the big-GPU phase" unless the credit terms
forbid that.

### 1.3 What $200 / $300 / $500 buys (GPU-hours)

| GPU / rate used | $200 | $300 | $500 |
|---|---|---|---|
| g4dn.xlarge T4 spot $0.23 | 870 h | 1,300 h | 2,170 h |
| g4dn.xlarge T4 on-demand $0.526 | 380 h | 570 h | 950 h |
| g5.xlarge A10G spot $0.50 | 400 h | 600 h | 1,000 h |
| g5.xlarge A10G on-demand $1.006 | 199 h | 298 h | 497 h |
| g6.xlarge L4 spot $0.45 | 440 h | 670 h | 1,110 h |
| g6.xlarge L4 on-demand $0.805 | 248 h | 373 h | 621 h |
| g6e.xlarge L40S spot $1.32 | 150 h | 230 h | 380 h |
| g6e.xlarge L40S on-demand $1.861 | 107 h | 161 h | 269 h |
| p4d spot $8.91/node (= 8 A100-40GB) | 22 node-h = 180 GPU-h | 34 node-h = 270 GPU-h | 56 node-h = 450 GPU-h |
| p4d on-demand $21.96/node | 9 node-h = 73 GPU-h | 14 node-h = 109 GPU-h | 23 node-h = 182 GPU-h |
| p5 spot ~$21/node (= 8 H100) | 9.5 node-h = 76 GPU-h | 14 node-h = 114 GPU-h | 24 node-h = 190 GPU-h |
| p5 on-demand $55.04/node | 3.6 node-h = 29 GPU-h | 5.5 node-h = 44 GPU-h | 9.1 node-h = 73 GPU-h |
| Vast.ai A100 80 GB ~$1.05 | 190 h | 285 h | 475 h |
| Vast.ai H100 SXM ~$2.20 | 91 h | 136 h | 227 h |
| RunPod Community A100 80 GB $1.19-1.39 | 145-170 h | 215-250 h | 360-420 h |
| RunPod Community H100 SXM $2.69 | 74 h | 112 h | 186 h |

Remember to subtract storage, EBS and egress (see §4): budget ~$15-40 of any credit pool for that.

---

## 2. Feasibility matrix

### 2.0 Reference hardware numbers used for the estimates

| Device | VRAM / memory | Mem BW | Dense FP16/BF16 tensor TFLOPS (nominal) | Planning "sustained" TFLOPS for training | bf16? |
|---|---|---|---|---|---|
| RTX 2060 (Turing) | 6 GB | 336 GB/s | ~52 (FP16 accumulate) / ~26 (FP32 acc.) | ~8-12 | no (fp16 only) |
| T4 (g4dn) | 16 GB | 320 GB/s | 65, but 70 W power-throttled | ~10-15 | no (fp16 only) |
| A10G (g5) | 24 GB | 600 GB/s | 70 | ~20-28 | yes |
| L4 (g6) | 24 GB | 300 GB/s | 121 | ~20-30 (bandwidth-bound in practice) | yes, FP8 |
| L40S (g6e) | 48 GB | 864 GB/s | 362 | ~90-120 | yes, FP8 |
| A100 SXM | 40/80 GB | 1.6-2.0 TB/s | 312 | ~100-130 | yes |
| H100 SXM | 80 GB | 3.35 TB/s | 989 | ~250-350 | yes, FP8 |
| Apple M4 (10-core GPU) | 16-32 GB unified (~70-75% usable for GPU) | 120 GB/s | ~4.3 FP32 (fp16 ~same rate, no tensor cores) | ~1.5-2 | bf16 supported in MLX/MPS, no HW speedup |
| Apple M4 Pro (20-core GPU) | 24/48/64 GB unified | 273 GB/s | ~9.2 FP32 ([flopper.io](https://flopper.io/gpu/apple-m4-pro)) | ~3-4 | same |
| Apple M5 Pro (new mini, Aug 2026) | up to 64 GB | 307 GB/s | GPU "neural accelerators" per core; Apple claims ~4x peak AI compute vs M4 generation; MLX prefill ~4x faster measured | ~10-15 (est., little training data yet) | yes |

Method for LoRA/QLoRA token throughput: FLOPs/token ~ 4N (forward + input-gradient backward, base
weights frozen) to 6N (with gradient checkpointing) where N = base params, plus 4-bit dequant
overhead; divide the sustained TFLOPS above by that. Cross-checked against: Unsloth's README T4
benchmarks (Llama-2-7B QLoRA 1.95x faster than HF on a free Colab T4; TinyLlama 1.1B one Alpaca
epoch = 84 min on a T4) ([unsloth.ai](https://unsloth.ai/tinyllama-gguf)); vendor reports that H100
is ~2-2.5x A100 for LoRA/QLoRA ([VESSL](https://vessl.ai/en/blog/lora-finetuning-cost-a100-h100-b200),
[Thunder Compute](https://www.thundercompute.com/blog/nvidia-a100-vs-h100)); MLX LoRA README
(Llama-7B ~475 tok/s on M2 Ultra, Mistral-7B ~250 tok/s on M1 Max 32 GB with 4 LoRA layers)
([mlx-examples/lora](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md)).

### 2.1 (i) Gradient-boosted trees on tabular + embedding features

| Hardware | Feasible? | Time for 2-5M rows x 200-500 features (LightGBM/XGBoost) |
|---|---|---|
| CPU (any 8+ core box, incl. M4 Pro) | yes | 2-15 min per fit; full 50-trial sweep in an afternoon |
| RTX 2060 | yes (XGBoost/LightGBM GPU hist; 6 GB holds ~5M x 300 fp32 = 6 GB -> use 16-bit bins or sample) | ~1-3 min per fit |
| Mac mini M4/M4 Pro | yes (CPU; LightGBM has no Metal backend, XGBoost no MPS) | 3-15 min per fit |
| AWS | overkill; a c7i.4xlarge ($0.71/h) or the same GPU box you use for phase 2 | minutes |

This is the baseline and the thing that will most likely still be in production. Author/follower
features, posting time, media type, text length, language, and PCA-reduced text/image embeddings
usually capture most of the predictable variance in engagement; the multimodal nets are fighting for
the residual.

### 2.2 (ii) 100-400M-param multimodal fusion transformer over frozen encoders, 1-5M posts

Design assumption (strongly recommended): encoders (SigLIP 2 / CLIP, text encoder, video encoder)
run **once** to produce cached embeddings (§2.4); the trainable fusion model sees ~10-200 embedding
tokens per post, not pixels. Then training cost is small everywhere:

- Compute per example (est.): 6 x 200M params x ~32 tokens (pooled image + 8 video frames + ~16
  text-chunk tokens + tabular tokens) ~ 4e10 FLOPs. 5M posts x 3 epochs = 15M examples ~ 6e17 FLOPs.
  If you instead keep full patch/token sequences (~300 tokens/post) it is ~10x that.

| Hardware | Fits? | Wall time for 5M posts x 3 epochs (pooled-token design) | Same, with ~300-token sequences |
|---|---|---|---|
| RTX 2060 6 GB | yes up to ~250M params with fp32 AdamW (16 B/param); 400M needs 8-bit Adam (bitsandbytes works on Turing) or Adafactor | ~15-30 h (est.; often dataloader-bound: 15M x ~10 KB embeddings/epoch = 150 GB read -> use memmapped fp16 arrays on NVMe) | 4-8 days -> impractical |
| Mac mini M4 Pro 48/64 GB (MPS or MLX) | yes; all ops are standard (Linear/LayerNorm/SDPA) so PyTorch MPS works | ~30-60 h (est.) | ~2 weeks -> impractical; subsample |
| T4 (g4dn.xlarge spot $0.23) | yes | ~10-20 h = $3-5 | 4-6 days = $25-35 |
| A10G (g5.xlarge spot $0.50) / L4 (g6.xlarge spot $0.45) | yes, bf16 | ~5-10 h = $3-5 | ~2-3 days = $25-35 |
| L40S (g6e spot $1.32) | yes | ~2-3 h = $3-4 | ~12-18 h = $16-24 |
| A100 80 GB (Vast $1.05) | yes | ~1-2 h = $2 | ~8-12 h = $10-13 |

Verdict: feasible on every option including the 2060 and the Mac, provided embeddings are precomputed
and stored compactly. Spend the GPU budget on the embedding pass and on hyperparameter sweeps, not on
one big run.

### 2.3 (iii) LoRA / QLoRA fine-tuning a 7-8B LLM or 2-4B VLM as an engagement predictor

Workload: 500k-2M examples x (~512 text tokens + 1 image). With Qwen2.5-VL/Qwen3-VL at
`max_pixels` ~ 448x448 an image is ~256 visual tokens (variable-resolution; cap it), so ~768
tokens/example; 500k examples ~ 0.38B tokens, 2M examples ~ 1.5B tokens per epoch. One epoch is
normal for this kind of regression/pairwise head; most signal comes from the first 200-500k
examples, so budget the 500k case and treat 2M as an upside.

Estimated QLoRA (4-bit, r=16-32, grad checkpointing, Unsloth/PEFT) training throughput, tokens/s
including image tokens, with the frozen ViT forward amortised (cache ViT outputs if the vision tower
stays frozen -> roughly +15-25% throughput):

| GPU | 7-8B (Qwen2.5-VL-7B, Qwen3-VL-8B, Llama-3.1-8B) | 2-4B (Qwen3-VL-2B/4B, Qwen2.5-VL-3B, Gemma 3 4B, SmolVLM2-2.2B) | Notes |
|---|---|---|---|
| RTX 2060 6 GB | does not fit (see §3) | 2B fits at bs 1-2 @512+image, ~250-500 tok/s est.; 3-4B 4-bit only with short ctx, OOM-prone | fp16 only, no bf16; Unsloth supports Turing (CUDA cap 7.0+) |
| M4 Pro 48/64 GB (MLX / mlx-vlm) | fits; ~200-350 tok/s est. (M1 Max measured 250, M2 Ultra 475) | ~600-1,200 tok/s est. | MLX LoRA only; no Unsloth/bnb/flash-attn; see §2.5 |
| T4 16 GB (g4dn) | fits (10-14 GB with image); ~250-500 tok/s est. | ~700-1,400 tok/s | fp16 only; Qwen fp16 overflow risk; slowest path |
| A10G 24 GB (g5) | ~700-1,200 tok/s est. | ~2,000-3,500 tok/s | bf16; the realistic "AWS small" option |
| L4 24 GB (g6) | ~700-1,300 tok/s est. | ~2,000-3,500 tok/s | bandwidth-bound; roughly A10G-class for QLoRA |
| L40S 48 GB (g6e) | ~2,500-4,500 tok/s est.; can do 16-bit LoRA on 7B instead of QLoRA (faster, ~3,500-5,000) | ~7,000-12,000 tok/s | best single-GPU option *inside AWS* |
| A100 80 GB | ~3,000-6,000 tok/s (Unsloth-class); 16-bit LoRA ~5,000-8,000 | ~9,000-16,000 | |
| H100 80 GB | ~8,000-15,000 tok/s; 16-bit LoRA ~12,000-20,000 | ~20,000-40,000 | |

Hours and dollars per epoch (mid-range of the estimates; multiply by 1.3-1.5 for eval, restarts and
spot interruptions):

| GPU | 7-8B, 500k ex (0.38B tok) | 7-8B, 2M ex (1.5B tok) | 3-4B, 500k ex | 3-4B, 2M ex |
|---|---|---|---|---|
| T4 spot $0.23 | ~290 h (12 days) = $67 | ~1,150 h = not practical | ~105 h = $24 | ~420 h = $97 |
| A10G spot $0.50 | ~110 h (4.5 days) = $55 | ~440 h (18 days) = $220, impractical | ~40 h = $20 | ~160 h = $80 |
| L4 spot $0.45 | ~105 h = $47 | ~420 h = $190 | ~38 h = $17 | ~150 h = $68 |
| L40S spot $1.32 | ~30 h = $40 | ~120 h (5 days) = $160 | ~11 h = $15 | ~45 h = $60 |
| A100 80 GB Vast $1.05 / RunPod $1.19-1.39 | ~24 h = $25-35 | ~95 h (4 days) = $100-130 | ~9 h = $10-13 | ~35 h = $37-50 |
| H100 Vast $2.20 / RunPod $2.69 | ~10 h = $22-27 | ~40 h = $88-108 | ~3.5 h = $8-10 | ~14 h = $31-38 |
| p4d spot (8x A100-40GB) $8.91/node, DDP | ~3.5 node-h = $31 | ~13 node-h = $116 | ~1.3 node-h = $12 | ~5 node-h = $45 |
| p5 spot (8x H100) ~$21/node | ~1.4 node-h = $30 | ~5.5 node-h = $115 | ~0.5 node-h = $11 | ~2 node-h = $42 |
| M4 Pro Mac mini (MLX) | ~350-500 h = 2-3 weeks of wall clock | months; no | ~120-170 h (5-7 days) | ~3-4 weeks |

Verdict: the 7-8B VLM at 500k-2M examples is an A100/H100/L40S job; on A10G/L4 it is only
sensible for a 2-4B VLM or a <=200k-example 7B pilot. On the 2060 and on a Mac it is a <=100k-example
pilot of a 2-4B model, nothing more.

Caveats on the published anchors: no public source gives tokens/s for Qwen2.5-VL-7B LoRA; the only
VLM datapoint found is "~4-12 h for 10k images with LoRA on a 24 GB GPU"
([Labellerr](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/)), i.e. 0.25-0.7 examples/s at a
larger `max_pixels` than assumed here, which is consistent with the A10G row above
(~1-1.5 ex/s at 768 tok/ex). Treat the whole table as +/-2x until benchmarked.

### 2.4 (iv) Precomputing embeddings: 2M images + 200k videos

Per-image cost (vision tower only, fp16): CLIP ViT-L/14@224 ~0.16 TFLOP; SigLIP 2 So400m/16@256
~0.22 TFLOP; SigLIP 2 So400m/14@384 ~0.5 TFLOP; PE-Core-L/14@336 ~0.37 TFLOP; SigLIP 2 B/16@256
~0.04 TFLOP. Real pipelines run at 15-40% of peak because JPEG decode + resize on 4 vCPUs is often
the bottleneck (LAION measured 312 img/s per A100 for ViT-L/14 end-to-end, well under compute peak
([LAION-5B paper](https://arxiv.org/pdf/2210.08402))). Use NVIDIA DALI / nvjpeg or torchvision.io
GPU decode, or pre-resize to 384 px once, to keep the GPU fed.

Time per 1,000 images (est., batch 64-128, fp16, decode not bottlenecked):

| Device | ViT-L/14@224 or So400m@256 | So400m@384 | 2M images, So400m@256 | 2M images, So400m@384 |
|---|---|---|---|---|
| RTX 2060 | ~7-10 s (100-150 img/s) | ~20-30 s (35-50 img/s) | ~4-5 h | ~12-16 h |
| T4 (g4dn.xlarge) | ~6-10 s (100-170 img/s); 4 vCPU decode caps ~150-250 img/s | ~17-30 s | ~3.5-5 h = $1-3 spot | ~10-16 h = $2.5-4 spot |
| A10G (g5.xlarge) | ~2.5-4 s (250-400 img/s), decode-limited on 4 vCPU -> use g5.2xlarge | ~7-11 s | ~1.5-2.5 h | ~4-6 h = $2-3 spot |
| L4 (g6.xlarge) | ~2.5-4 s | ~7-10 s | ~1.5-2.5 h | ~4-6 h |
| L40S / A100 | ~0.8-1.5 s | ~2-4 s | <1 h | ~1.5-2.5 h |
| M4 Pro (MLX or Core ML; ANE helps for B-size models) | ~25-50 s (20-40 img/s) | ~70-120 s (8-15 img/s) | ~15-25 h | ~40-70 h |
| M4 base | ~60-100 s | ~3-4 min | ~35-55 h | several days |

Anchors: NVIDIA TAO lists SigLIP 2 So400m/16@256 TensorRT FP16 on A100 and H100
([NGC siglip_v2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/siglip_v2)); Chinese-CLIP
reports ViT-L/14 TensorRT FP16 on a T4 ([paper](https://arxiv.org/pdf/2211.01335)); OpenCLIP
community throughput gist ([gist](https://gist.github.com/TACIXAT/ecd4f636bf6af28cb69d641e29d7b362)).

Videos (200k, sample 8-16 frames each -> 1.6-3.2M frames):
- Decoding dominates: ffmpeg keyframe-seek + scale for a 15-60 s clip is ~0.3-1 s on 4 vCPU; 200k
  clips ~ 20-55 CPU-core-hours -> 2-4 h on a 16-vCPU box, or run it on the 2060 host / Mac mini
  (12-14 cores) over a night. Do this once at ingest and store the sampled frames, never the videos.
- Encoding options per clip: (a) per-frame SigLIP on 8-16 frames + mean/attention pool = 8-16x the
  image cost (simplest, same model as images; 200k clips = 1.6-3.2M frames -> add the "2M images"
  column again); (b) InternVideo2-1B @224, 8 frames ~2 TFLOP/clip; VideoMAE v2 ViT-g 16 frames
  ~2-3 TFLOP/clip; V-JEPA 2 ViT-L @256, 16 frames ~1.2 TFLOP/clip. On an A10G/L4 at ~25 TFLOPS
  effective that is 8-20 clips/s -> 200k clips in 3-7 h (~$2-4 spot); T4 ~2.5x longer; 2060 similar
  to T4 but 1B models at fp16 (2 GB) + 1-2k-token activations fit in 6 GB only at batch 2-4; M4 Pro
  ~1-2 clips/s -> 30-55 h.

Verdict: embedding precompute is cheap on any NVIDIA GPU (single-digit dollars on spot) and
feasible-but-slow (one to several days) on the Mac. The real cost is data movement (§4).

### 2.5 Mac mini specifics (what MLX / PyTorch-MPS can and cannot do)

- Memory: unified; by default MLX/MPS can wire ~70-75% of RAM for the GPU; macOS + apps take 4-6 GB.
  A 16 GB M4 is limited to ~1.5-3B models for training ([HF blog, 16 GB Mac MPS vs MLX](https://huggingface.co/blog/harshitkgupta/fine-tuning-coding-agents-on-mac-pytorch-mps-mlx)); 24 GB
  M4 Pro handles 7B 4-bit QLoRA (~8-10 GB); 48/64 GB handles 7B 16-bit LoRA (~19-22 GB) and 14B 4-bit.
- Speed: no tensor cores on M4/M4 Pro; fp16/bf16 matmul runs at roughly fp32 shader rate (~9 TFLOPS
  on M4 Pro vs 70 on an A10G, 312 on an A100). Measured MLX LoRA: 7B at 250 tok/s (M1 Max) - 475 tok/s
  (M2 Ultra); expect M4 Pro ~250-350 tok/s. The Aug-2026 M5 Pro mini adds per-core neural
  accelerators (Apple claims ~4x peak AI compute; measured ~4x MLX prefill), so if the sponsor is
  *buying* a Mac, buy the M5 Pro 64 GB, not an M4.
- MLX (recommended on Mac): `mlx_lm.lora` for LLMs; `mlx-vlm` supports LoRA on Qwen2-VL/Qwen2.5-VL/
  Qwen3-VL, SmolVLM, Gemma 3, Pixtral etc.; MLX has its own 4/8-bit affine quantisation (replaces
  bitsandbytes), fused SDPA kernels (not FlashAttention-parity), lazy evaluation; no DeepSpeed, no
  vLLM, no Unsloth (Unsloth says Apple Silicon/MLX support "is in the works"; Unsloth Studio only does
  MLX/GGUF inference on Mac) ([Unsloth requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements)).
- PyTorch MPS: works for standard nets (fusion model, encoders); **bitsandbytes 4/8-bit is CUDA-only**
  (an MPS backend PR exists, with failing optimizer tests, unmerged as of this search
  ([bnb PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853))); no
  FlashAttention-2, no xformers; SDPA exists but is slower and has had correctness bugs in some
  releases (PyTorch 2.10 MPS two-pass attention bug noted in
  [this PR](https://github.com/sh3ngj/YuE/pull/1)); FlexAttention arrived on MPS in PyTorch 2.13 but is
  marked API-unstable; some ops still fall back to CPU; `pin_memory`/`num_workers>0` waste memory on
  unified-memory Macs; the MPS high/low watermark ratios need tuning to use all memory. In the HF
  16 GB comparison MPS fp16 was 2-6x faster per step than MLX 4-bit at 0.5-1.5B (dequant cost), but
  MLX fit 2x larger models and scaled to 4k context.
- Core ML / ANE: good for *inference* of B-size encoders (SigLIP-B, CLIP-B) at 100-200 img/s, not for
  training.
- Conclusion: a Mac mini is a fine phase-1/phase-2 machine and a usable 2-4B VLM pilot machine; it is
  not a 7B x 500k-example training machine.

---

## 3. Memory arithmetic

QLoRA 7B at 512 context (Llama-3.1-8B / Qwen2.5-7B class, NF4 base, LoRA r=16 on all linear
layers, AdamW, gradient checkpointing):
- 4-bit base weights: 7-8B x 0.5 B ~ 3.5-4 GB, plus non-quantised embeddings/LM head/norms
  (Llama 128k vocab untied: ~1 GB bf16; Qwen 152k vocab: ~2.2 GB bf16 unless also quantised).
- LoRA params ~40-80M x 16 B (fp32 weight + grad + 2 Adam states) ~0.6-1.3 GB.
- Activations at bs=1, 512 tokens with checkpointing: ~0.2-0.4 GB; logits 512 x 128-152k vocab in
  fp32 x 2-3 copies ~0.8-1.4 GB unless chunked (Unsloth chunks); scale linearly with batch size.
- CUDA context + allocator slack ~0.6-1 GB.
- Total: ~6-7 GB at bs 1-2; ~8-11 GB at bs 8-16; this matches Unsloth's published minimums (7B 5 GB,
  8B 6 GB at 4-bit; 19/22 GB at 16-bit LoRA) ([Unsloth requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements))
  and the community's "8-10 GB in practice" ([Spheron VRAM guide](https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/)).
- 7B **VLM** (Qwen2.5-VL-7B): add the ViT (675M, 1.35 GB bf16 or ~0.4 GB 4-bit) and 256-1,280
  visual tokens per image -> 10-14 GB at bs 2-4 with `max_pixels` capped near 448x448.

What fits where:

| VRAM | Honest answer |
|---|---|
| 6 GB (RTX 2060) | 7B QLoRA technically scrapes Unsloth's 5 GB floor at bs=1/short ctx but is OOM-prone and fp16-only (no bf16 on Turing) -> treat as not viable. Viable: 0.5-2B QLoRA/LoRA (Qwen2.5-0.5B/1.5B, Qwen3-1.7B, SmolVLM2-256M/500M/2.2B-4bit), 3B 4-bit at bs=1 and <=512 ctx is tight; fusion models <=250M params with fp32 AdamW (16 B/param = 4 GB) or <=400M with 8-bit Adam; inference of any <=3B 4-bit model; batch-64 embedding inference with 1B encoders. |
| 16 GB (T4) | 7-8B QLoRA incl. VLM at bs 2-4; 14B QLoRA text-only at bs 1; no 16-bit LoRA above ~3B; fp16 only. |
| 24 GB (A10G / L4 / RTX 3090-4090) | QLoRA up to 27-32B (Unsloth min 22-26 GB, bs 1, short ctx -> painful); comfortable: 7-14B QLoRA at bs 8-16 with 512 tok + image; 16-bit LoRA up to 7-9B (19-24 GB, bs 1-2); full fine-tune <=1B (16 B/param); any fusion model. |
| 48 GB (L40S / A6000 / A40) | QLoRA up to ~70B at bs 1 (41 GB floor, no headroom); comfortable 27-32B QLoRA; 16-bit LoRA up to 14B comfortably, 27B tight (64 GB floor -> no); full FT <=2B with 8-bit Adam; 7B VLM QLoRA at bs 16-32 with long image sequences. |
| 80 GB (A100/H100) | 16-bit LoRA 27-32B; QLoRA 70B at bs 4-8; 7B full FT only with 8-bit Adam + bf16 (~42 GB + activations) or FSDP. |
| Mac unified 24 / 48 / 64 GB | ~17 / 36 / 48 GB usable: 7B 4-bit LoRA (MLX) / 7B 16-bit LoRA or 14B 4-bit / 32B 4-bit LoRA at bs 1. Speed, not memory, is the constraint. |

---

## 4. Storage and bandwidth for a 2M-post multimodal dataset

Dataset size (assumptions: 2M images at ~100 KB; 200k videos, keep 16 sampled JPEG frames at
~40-60 KB each instead of the source video; text + metadata ~2 KB/post):

| Component | Size |
|---|---|
| Images (2M x 100 KB) | ~200 GB |
| Video frames (200k x 16 x ~50 KB) | ~160 GB (source videos at ~5 MB avg would be ~1 TB: do not keep) |
| Text + metadata + labels (parquet) | ~4-6 GB |
| Image embeddings (2M x 1152-d fp16) | ~4.6 GB per encoder |
| Video embeddings (200k x 16 x 1152 fp16 / pooled) | ~7.4 GB / ~0.5 GB |
| Text embeddings (2M x 1024 fp16) | ~4 GB |
| Total raw ~370 GB; derived ~20 GB | |

AWS costs (us-east-1, list, Nov 2026 ([CloudZero](https://www.cloudzero.com/blog/s3-pricing/),
[Upstash S3 explainer](https://upstash.com/blog/aws-s3-pricing-explained-2026),
[OpsLyft](https://www.opslyft.com/blog/amazon-s3-pricing-per-gb))):
- S3 Standard $0.023/GB-month -> 400 GB = $9.2/month; S3 Standard-IA $0.0125 -> $5/month; Glacier
  Instant $0.004 -> $1.6/month (good for archiving raw images after embeddings exist).
- S3 requests: PUT $0.005/1k -> 2.2M objects = $11 (pack into ~1 GB WebDataset tar shards / parquet
  with image bytes to make this $0 and to stream efficiently); GET $0.0004/1k -> $0.9 per full read.
- S3 -> EC2 in the same region: free. S3 -> internet / another cloud: $0.09/GB after 100 GB free ->
  moving the 400 GB raw set to Vast/RunPod once = ~$27-36; moving just the ~20 GB of embeddings = ~$2.
  Plan the cross-cloud step so only embeddings + labels (not pixels) leave AWS, or stage the raw
  corpus on Cloudflare R2 ($0.015/GB-month, zero egress) / Backblaze B2 ($6/TB-month) if you expect
  to bounce between AWS and marketplaces.
- EBS gp3 $0.08/GB-month (3k IOPS, 125 MB/s included) -> a 500 GB working volume = $40/month; a 1 TB
  volume = $80/month. Cheaper: use the free local NVMe on the GPU instances (g4dn.xlarge 125 GB,
  g5.xlarge / g6.xlarge / g6e.xlarge 250 GB, g5.2xlarge 450 GB) as a cache and keep the source of
  truth in S3; keep a small 100 GB gp3 root/checkpoint volume ($8/month).
- Marketplace storage: RunPod network volume $0.07/GB-month (<1 TB), container/volume disk
  $0.10/GB-month; Vast.ai per-host disk typically $0.10-0.20/GB-month and is lost when the host goes
  away -> keep checkpoints in S3/R2.
- Ingest bandwidth: 2M images x 100 KB = 200 GB; at 50-100 MB/s with 32-64 concurrent fetches that is
  ~1 h of network, but X's CDN rate limits will dominate -> plan days, run it from the 2060 host or a
  $0.04/h t3.small, and write shards straight to S3.
- Rule of thumb for the budget: ~$10-15/month S3 + ~$8-40/month EBS over a 3-month project =
  $55-165 of a $300 credit pool if you are careless, ~$40 if you use local NVMe and tiered S3.

---

## 5. Open-weight encoders (2026): sizes and licences

Image
| Model | Sizes (vision tower) | Licence | Notes |
|---|---|---|---|
| SigLIP 2 (Google, Feb 2025) | B 86M, L 303M, So400m ~400M (428M), g ~1B; fixed-res 224/256/384/512 + NaFlex variable-res | Apache-2.0 ([HF card](https://huggingface.co/google/siglip2-so400m-patch16-512), [paper](https://huggingface.co/papers/2502.14786)) | Best default: multilingual text tower, strong zero-shot and dense features; So400m@256 or @384 for this task |
| CLIP ViT-L/14 (OpenAI; OpenCLIP/DataComp/LAION variants) | 304M vision (428M total) | MIT | Widely used in engagement/aesthetics work; weaker than SigLIP 2 / PE-Core |
| DINOv2 (Meta, 2023) | S 22M / B 86M / L 300M / g 1.1B | Apache-2.0 | Pure-visual features, no text alignment; good for image-quality/aesthetic signals |
| DINOv3 (Meta, Aug 2025) | ViT-S/S+/B/L/H+ distilled from ViT-7B (6.7B); ConvNeXt variants | Custom "DINOv3 License" (gated; permits commercial use with Meta's restrictions) ([HF card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)) | Best dense/visual features; check licence if the model is ever shipped |
| PE-Core (Meta Perception Encoder, Apr 2025) | B/16 ~90M, L/14 ~320M, G/14 ~1.88B | Apache-2.0 ([HF card](https://huggingface.co/facebook/PE-Core-L14-336)) | CLIP-style; PE-Core-L@336 is a strong, cheap option; G is heavy |

Video
| Model | Sizes | Licence | Notes |
|---|---|---|---|
| InternVideo2 (OpenGVLab, 2024) | Stage-2 1B (224p, 4-8 frames), 6B; CLIP-S variants | Official HF repos tagged Apache-2.0; some mirrors MIT; check repo ([HF](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4)) | Video-text aligned; good retrieval embeddings |
| VideoMAE v2 (OpenGVLab, 2023) | ViT-g 1B; distilled ViT-S/B | Model cards CC-BY-NC-4.0 (non-commercial) although the HF org repo is tagged Apache-2.0 ([HF](https://huggingface.co/OpenGVLab/VideoMAEv2-Base), [model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md)) | Licence risk for commercial use |
| V-JEPA 2 (Meta, Jun 2025) | ViT-L 300M, ViT-H 600M, ViT-g 1B (256px; g also 384px); 16-64 frames | Code MIT (some files Apache-2.0); HF checkpoints facebook/vjepa2-* ([GitHub](https://github.com/facebookresearch/vjepa2)) | Self-supervised motion/appearance features; no text alignment |
| Qwen2.5-VL / Qwen3-VL native video | 3B/7B/72B; Qwen3-VL 2B/4B/8B/32B + 30B-A3B/235B-A22B MoE | Qwen2.5-VL-7B Apache-2.0, 3B "Qwen Research" licence; Qwen3-VL all Apache-2.0 | Use the VLM itself as the video encoder in phase 3; 2 fps dynamic sampling |
| Practical alternative | per-frame SigLIP 2 + temporal pooling | Apache-2.0 | Cheapest and often within noise of dedicated video encoders for short social clips |

Audio
| Model | Size | Licence |
|---|---|---|
| Whisper large-v3 / large-v3-turbo (speech-to-text) | 1.55B / 809M | MIT ([Simon Willison's note](https://www.simonwillison.net/2024/Oct/1/whisper-large-v3-turbo-model)) |
| CLAP (LAION clap-htsat-fused/unfused) | ~150M | Apache-2.0 ([HF](https://huggingface.co/laion/clap-htsat-unfused)) ; Microsoft CLAP is MIT |
| BEATs (Microsoft unilm) | ~90M | MIT (unilm repo) |

Text
| Model | Size | Licence | Notes |
|---|---|---|---|
| Qwen3-Embedding | 0.6B (1024-d), 4B (2560-d), 8B (4096-d); 32k ctx, 100+ languages | Apache-2.0 ([paper](https://arxiv.org/abs/2506.05176v1)) | Recommended: X is multilingual; 0.6B is cheap enough for 2M posts on any GPU |
| ModernBERT | base 149M, large 395M; 8k ctx; English-centric | Apache-2.0 ([HF](https://huggingface.co/answerdotai/ModernBERT-base)) | Good trainable text tower for the fusion model if English-only |
| multilingual-e5-large / -instruct | ~560M (XLM-R large) | MIT | Solid multilingual alternative; bge-m3 (568M, MIT) similar |

VLM bases for phase 3: Qwen3-VL-2B/4B/8B (Apache-2.0) > Qwen2.5-VL-7B (Apache-2.0) > Gemma 3 4B/12B
(Gemma Terms of Use; Gemma 4 moved to Apache-2.0 in 2026 per secondary sources) > SmolVLM2
256M/500M/2.2B (Apache-2.0, tiny, good for the 2060/Mac pilot).

---

## 6. Serving cost per 1,000 predictions

Assumptions: one prediction = 1 image + ~512 text tokens; regression/pairwise head means prefill
only (no autoregressive decode); batched serving with vLLM (VLM) or ONNX/TensorRT (fusion); AWS
on-demand prices from §1 (spot/marketplace is ~half).

| Model | GPU/CPU | Throughput (est.) | Cost per 1,000 predictions |
|---|---|---|---|
| 7B QLoRA VLM (merged LoRA, bf16 or AWQ-int4, Qwen2.5-VL-7B @448px) | A10G g5.xlarge $1.006/h | ~2.5-4 pred/s (768-token prefill ~0.2-0.3 s + ViT) | ~$0.07-0.11 (spot ~$0.04-0.05) |
| | L4 g6.xlarge $0.805/h | ~2.5-4 pred/s | ~$0.06-0.09 |
| | A100 80 GB $1.05-1.39/h (Vast/RunPod) | ~13-20 req/s measured for Qwen2.5-VL-7B image inputs on A100 in vLLM ([vLLM issue #24728](https://github.com/vllm-project/vllm/issues/24728)) | ~$0.02-0.03 |
| | CPU c7i.4xlarge (16 vCPU, AMX) $0.714/h | ~0.1-0.2 pred/s (llama.cpp Q4, 768-token prefill 5-10 s) | ~$1-2 -> not viable beyond a demo |
| 300M fusion model + frozen SigLIP 2 So400m@384 + Qwen3-Embedding-0.6B | A10G g5.xlarge | ~30-50 pred/s (encoders dominate: ~0.7 TFLOP/pred) | ~$0.006-0.01 |
| | L4 g6.xlarge | ~30-50 pred/s | ~$0.005-0.008 |
| | CPU c7i.xlarge (4 vCPU) $0.1785/h | ~1-2 pred/s with ONNX int8 | ~$0.025-0.05 |
| | CPU c7i.xlarge with SigLIP 2 B/16@256 instead of So400m | ~8-15 pred/s | ~$0.003-0.006 |
| GBDT on cached features | CPU | 10k+ pred/s | <$0.0001 |

So the 7B VLM costs ~10x the fusion model on the same GPU and ~50x on CPU; at X-like volumes the
VLM is a teacher/offline scorer (or a re-ranker on the top-k) and the fusion model / GBDT is the thing
that serves. Distilling VLM scores into the fusion model is the standard pattern. Sources for prices:
[Holori c7i.xlarge](https://calculator.holori.com/aws/ec2/c7i.xlarge/us-east-1),
[Cerebrium A10 vs H100 inference](https://cerebrium.ai/blog/running-llama-3-8b-with-tensorrt-llm-on-serverless-gpus).

---

## 7. Recommended three-phase budget plan

Guiding principle: each phase must produce a model and an offline metric (Spearman / NDCG@k on a
time-split holdout) before the next phase gets money; the expensive phase is only justified if phase
2 shows that image/video content carries signal beyond author + timing features.

### Phase 1 — Baseline on the 2060 / CPU ($0 cloud)
- Build the dataset: pull 2-5M posts with labels (likes/reposts/replies/impressions at fixed
  horizons), pack images + sampled video frames into WebDataset shards, write parquet metadata.
  Runs on the 2060 host; 200-360 GB on a local disk.
- Precompute embeddings on the 2060: SigLIP 2 So400m@256 for 2M images (~5 h), per-frame SigLIP
  for 200k x 8 frames (~4 h), Qwen3-Embedding-0.6B for 2M texts (~2-3 h), Whisper-turbo transcripts
  for videos with speech (optional; ~0.5-1 s per clip -> run overnight for a 50k subset).
- Train LightGBM/XGBoost on tabular + PCA-64 of each embedding; pairwise/lambdarank objective for
  "which of two posts from the same author did better" plus a log-engagement regression.
- Output: baseline metrics, feature importances (tells you whether media features matter at all),
  frozen train/val/test splits. Time: 1-2 weeks of engineering, ~15-20 GPU-hours on the 2060.

### Phase 2 — Fusion model on A10G/L4 spot (~$40-80 of AWS credit)
- Upload shards to S3 (same region as compute; ~$10/month) or ship only the ~20 GB of embeddings.
- g5.xlarge / g6.xlarge spot ($0.45-0.50/h) or g5.2xlarge if decode-bound: re-embed with the final
  encoder choice if needed (2M images So400m@384 ~5 h = $2.5; video 200k clips ~5 h = $2.5), then
  train a 100-300M fusion transformer over cached embeddings: ~5-10 h per run x 5-8 sweep runs =
  40-80 GPU-h = $20-40. Optionally unfreeze the last 2-4 encoder blocks for one run (~3x cost).
- Storage/EBS over two months ~$20-30. Total ~$50-80.
- Output: a servable model (~$0.01/1k predictions on L4) and the uplift over phase 1 that decides
  whether phase 3 is worth it.

### Phase 3 — VLM LoRA on A100/H100 ($60-200 depending on budget)
- Model: Qwen3-VL-8B (or Qwen2.5-VL-7B) QLoRA r=32, regression + pairwise head, `max_pixels`
  capped at ~448x448, 512 text tokens, 1 epoch over a stratified 300-500k-example subset (hard pairs
  and tail events over-sampled); plus a Qwen3-VL-4B run as the cheap ablation.
- Where: Vast.ai verified A100 80 GB at ~$1.05/h or H100 at ~$2.20/h; RunPod Community as fallback.
  Inside AWS, g6e.xlarge (L40S 48 GB) spot at $1.32/h is the single-GPU choice (~30 h for 500k
  examples = ~$40); p4d spot ($8.91/h for 8x A100-40GB, ~3.5 h with DDP) is the fastest AWS route
  but needs multi-GPU plumbing and is interruption-prone.
- Budget: 2 x 50k-example pilot runs (2 x 3 h A100 = $7) + one 500k run on A100 (~24 h, $25-35) or
  H100 (~10 h, $22-27) + 4B ablation ($10) + scoring the 200k-post validation set and the full corpus
  for distillation (2M posts at ~15 pred/s on A100 = ~37 h = $40) + 30% slack = **$100-130**.
  Going to 2M training examples adds ~$75-100 on A100 or ~$65-80 on H100.
- Output: VLM scores on the full corpus; distil into the phase-2 fusion model (another ~$10 of
  A10G time) so that serving stays cheap.

Totals
| Budget | Plan |
|---|---|
| $200 | Phase 1 ($0) + Phase 2 ($50) + Phase 3 limited to 4B VLM on 500k or 7-8B on <=300k ($80-100) + storage ($20-30). All three phases fit if phase 3 runs on Vast/RunPod; AWS-only credits fit phase 2 + a g6e (L40S) 7B run on ~300k examples. |
| $300 | Full plan: phase 2 with sweeps ($70), phase 3 7-8B on 500k-1M + 4B ablation + full-corpus scoring ($130-170), storage ($30-40). |
| $500 | As above plus 7-8B on the full 2M examples ($100-130 extra on A100/H100) and a second seed / encoder ablation; or buy ~190 h of H100 at $2.20 and stop worrying about it. |

Approximate GPU-hour ledger for the $300 plan: 2060 ~20 h (free), A10G/L4 spot ~100-130 h
(~$55), L40S or A100 ~40-50 h (~$50-55 on Vast, ~$60 on g6e spot), inference/scoring ~40 h (~$40),
storage/egress ~$35. Total ~$180-200 with ~$100 of contingency for spot interruptions and respins,
which empirically consume 20-40% of a small budget.

### Plan B — only the Mac mini is available
- Which Mac matters: a base M4 16 GB is a phase-1 machine only (3B max, slow). An M4 Pro 48/64 GB
  does phases 1-2 fully and a 2-4B VLM pilot; the Aug-2026 M5 Pro mini (64 GB, 307 GB/s, neural
  accelerators, from $1,699) is 3-4x faster for matmul and is what to buy if buying new
  ([MacRumors 2024 vs 2026 mini](https://macrumors.com/guide/2024-vs-2026-mac-mini),
  [RedShark M5 Pro specs](https://www.redsharknews.com/mac-mini-m6-m5-pro-price-specs)).
- Phase 1: identical (CPU LightGBM; 12-14 performance/efficiency cores are plenty).
- Embeddings: prefer SigLIP 2 B/16@256 via MLX or Core ML (ANE) at ~100-200 img/s -> 2M images in
  3-6 h; So400m@256 via MLX at ~25-35 img/s -> 16-22 h (a weekend); video frames at 8/clip add
  1.6M frames; text with Qwen3-Embedding-0.6B in MLX ~100-200 posts/s -> 3-6 h. All overnight jobs.
- Fusion model: PyTorch MPS or MLX over cached embeddings, <=300M params, 5M posts x 3 epochs in
  roughly 1-3 days; this is fully feasible and is the main deliverable.
- VLM: `mlx-vlm` LoRA on Qwen3-VL-2B/4B or SmolVLM2-2.2B at 4-bit, 50-100k curated examples, ~1-3
  days per run at ~600-1,200 tok/s. Do not attempt 7-8B x 500k on an M4 (2-3+ weeks per epoch,
  thermal throttling in a mini); on an M5 Pro a 7B x 100-200k pilot becomes a multi-day but
  tolerable job.
- Serving: fusion model on the Mac at tens of predictions/s via Core ML; the VLM only as an offline
  scorer (~0.3-0.5 pred/s).
- Tooling to avoid on Mac: bitsandbytes, Unsloth, flash-attn/xformers, DeepSpeed, vLLM, NVIDIA DALI;
  use MLX quantisation, `mlx_lm` / `mlx-vlm`, Core ML Tools, and plain PyTorch MPS for the fusion net.

---

## Sources (consolidated)

AWS prices: [DoiT g6e.xlarge](https://www.doit.com/compute/compute/aws/us-east-1/g6e.xlarge) ·
[Holori g6e.xlarge](https://calculator.holori.com/aws/ec2/g6e.xlarge) ·
[DevZero g6e.xlarge](https://www.devzero.io/instances/aws/g6e.xlarge) ·
[DevZero g6.xlarge](https://www.devzero.io/instances/aws/g6.xlarge) ·
[Vantage g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge) ·
[Vantage g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge) ·
[DoiT g5.xlarge](https://www.doit.com/compute/compute/aws/us-east-1/g5.xlarge) ·
[DoiT g5.2xlarge](https://www.doit.com/compute/spot/us-east-1/g5.2xlarge) ·
[DoiT g4dn.xlarge](https://www.doit.com/compute/spot/us-east-1/g4dn.xlarge) ·
[DevZero g4dn.xlarge](https://www.devzero.io/instances/aws/g4dn.xlarge) ·
[DoiT p4d.24xlarge](https://compute.doit.com/spot/us-east-1/p4d.24xlarge) ·
[DoiT p5.48xlarge](https://www.doit.com/compute/spot/us-east-1/p5.48xlarge) ·
[Holori p5.4xlarge](https://calculator.holori.com/aws/ec2/p5.4xlarge) ·
[CloudOptimo price cuts](https://www.cloudoptimo.com/blog/aws-announces-significant-price-reductions-for-nvidia-gpu-ec2-instances/) ·
[OpsLyft AWS GPU pricing 2026](https://www.opslyft.com/blog/aws-gpu-pricing) ·
[Thunder Compute EC2 GPU guide](https://www.thundercompute.com/blog/ec2-gpu-instances) ·
[Spheron AWS H100 pricing 2026](https://www.spheron.network/blog/aws-h100-pricing-2026/) ·
[Holori c7i.xlarge](https://calculator.holori.com/aws/ec2/c7i.xlarge/us-east-1) ·
[Holori c7a.xlarge](https://calculator.holori.com/aws/ec2/c7a.xlarge/us-east-1)

Marketplaces: [Lambda pricing (effective 2026-11-16)](https://lambda.com/pricing) ·
[Thunder Compute on Lambda](https://www.thundercompute.com/blog/lambda-labs-vs-thunder-compute) ·
[RunPod pricing](https://www.runpod.io/pricing) ·
[computeprices.com RunPod](https://computeprices.com/providers/runpod) ·
[computeprices.com Vast](https://computeprices.com/providers/vast) ·
[JouleMath GPU rental prices Oct 2026](https://joulemath.com/blog/gpu-rental-prices-h100-h200-a100-rtx-4090-october-2026/) ·
[Vast.ai review 2026](https://www.gpunex.com/blog/vast-ai-review-2026/)

Spot reliability: [Thunder Compute spot availability](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability) ·
[Spheron spot GPU 2026](https://www.spheron.network/blog/spot-gpu-instances-2026-savings-and-what-breaks/)

Training memory/throughput: [Unsloth requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements) ·
[Unsloth TinyLlama/T4 benchmarks](https://unsloth.ai/tinyllama-gguf) ·
[Spheron VRAM guide 2026](https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/) ·
[stack-junkie QLoRA 8B peak 6.6 GB](https://www.stack-junkie.com/blog/lora-fine-tuning-consumer-gpu) ·
[VESSL A100 vs H100 vs B200 LoRA](https://vessl.ai/en/blog/lora-finetuning-cost-a100-h100-b200) ·
[Thunder Compute A100 vs H100](https://www.thundercompute.com/blog/nvidia-a100-vs-h100) ·
[Labellerr Qwen2.5-VL fine-tune](https://www.labellerr.com/blog/fine-tune-qwen-2-5-vl/) ·
[Qwen2.5-1.5B on 8 GB profiling (arXiv 2509.12229)](https://arxiv.org/abs/2509.12229v1)

Mac: [mlx-examples LoRA README](https://github.com/ml-explore/mlx-examples/blob/main/lora/README.md) ·
[MLX discussion #1647](https://github.com/ml-explore/mlx/discussions/1647) ·
[HF blog: 16 GB Mac MPS vs MLX](https://huggingface.co/blog/harshitkgupta/fine-tuning-coding-agents-on-mac-pytorch-mps-mlx) ·
[bitsandbytes MPS PR #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853) ·
[PyTorch 2.13 FlexAttention on MPS](https://ecorpit.com/pytorch-2-13-flexattention-apple-silicon-12x-speedup-2026/) ·
[Scalastic Apple Silicon vs CUDA](https://scalastic.io/en/apple-silicon-vs-nvidia-cuda-ai-2025/) ·
[flopper.io M4 Pro](https://flopper.io/gpu/apple-m4-pro) ·
[M5 Pro vs M4 Max ML benchmarks](https://dev.to/valesys/gpu-accelerated-mldl-performance-on-macbook-pro-m5-pro-vs-m4-max-feasibility-and-benchmarks-for-2ka3) ·
[MacRumors 2024 vs 2026 Mac mini](https://macrumors.com/guide/2024-vs-2026-mac-mini) ·
[RedShark M6/M5 Pro mini](https://www.redsharknews.com/mac-mini-m6-m5-pro-price-specs)

Encoders: [SigLIP 2 paper](https://huggingface.co/papers/2502.14786) ·
[google/siglip2-so400m-patch16-512](https://huggingface.co/google/siglip2-so400m-patch16-512) ·
[DINOv3 ViT-L card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) ·
[PE-Core-L14-336](https://huggingface.co/facebook/PE-Core-L14-336) ·
[timm PE-Core-B-16](https://huggingface.co/timm/PE-Core-B-16) ·
[V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2) ·
[InternVideo2-Stage2_1B](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4) ·
[VideoMAEv2-Base card](https://huggingface.co/OpenGVLab/VideoMAEv2-Base) ·
[VideoMAEv2 model zoo](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.md) ·
[Qwen3-Embedding paper](https://arxiv.org/abs/2506.05176v1) ·
[ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) ·
[ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) ·
[laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused) ·
[Whisper large-v3-turbo note](https://www.simonwillison.net/2024/Oct/1/whisper-large-v3-turbo-model) ·
[SmolVLM2-2.2B card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/blob/main/README.md) ·
[Gemma 3 4B card](https://huggingface.co/google/gemma-3-4b-it) ·
[Gemma 4 Apache-2.0 note](https://www.noze.it/en/insights/gemma-new-release/) ·
[Qwen3-VL overview](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-VL.md)

Embedding/inference throughput: [LAION-5B paper (312 img/s/A100 ViT-L/14)](https://arxiv.org/pdf/2210.08402) ·
[NGC SigLIP 2 TensorRT numbers](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/siglip_v2) ·
[Chinese-CLIP T4 TensorRT](https://arxiv.org/pdf/2211.01335) ·
[OpenCLIP throughput gist](https://gist.github.com/TACIXAT/ecd4f636bf6af28cb69d641e29d7b362) ·
[vLLM multimodal benchmark on A100 (#24728)](https://github.com/vllm-project/vllm/issues/24728) ·
[Cerebrium Llama-3-8B A10 vs H100](https://cerebrium.ai/blog/running-llama-3-8b-with-tensorrt-llm-on-serverless-gpus)

Storage: [CloudZero S3 pricing 2026](https://www.cloudzero.com/blog/s3-pricing/) ·
[Upstash S3 pricing explained 2026](https://upstash.com/blog/aws-s3-pricing-explained-2026) ·
[OpsLyft S3 per GB](https://www.opslyft.com/blog/amazon-s3-pricing-per-gb) ·
[n2ws AWS storage costs](https://n2ws.com/blog/aws-cost-optimization/understanding-aws-storage-costs)
