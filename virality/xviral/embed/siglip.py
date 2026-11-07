"""SigLIP-2 image and frame embeddings (DATA_PIPELINE.md 7.7).

The 1024-d post embedding compresses visual detail; Phoenix compensates with
per-post memory, which a surrogate scoring unseen posts does not have.
SigLIP-2 So400m (Apache-2.0) supplies per-image and per-frame vectors.
"""

from __future__ import annotations

import io

import numpy as np


class SiglipEncoder:
    def __init__(self, model_id: str = "google/siglip2-so400m-patch14-384", device: str = "cuda", batch_size: int = 64):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise RuntimeError("install the 'encoders' extra (torch, transformers) on the GPU box") from e
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    def encode(self, images: list[bytes]) -> np.ndarray:
        from PIL import Image

        outs = []
        for i in range(0, len(images), self.batch_size):
            ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in images[i:i + self.batch_size]]
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                f = self.model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            outs.append(f.float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, 0), dtype=np.float32)
