"""Modal download entry for SCAIL-2 (ComfyUI native path).

Run:
  modal run download.py::download

Fetches the ComfyUI-format weights the official SCAIL-2 workflow needs onto the
shared ``models`` volume, laid out as ComfyUI model dirs under /models/comfyui:

  diffusion_models/  SCAIL-2 14B DiT (Comfy-Org)
  loras/             SCAIL-2 DPO lora + Lightx2v 8-step distill lora
  vae/               Wan2.1 VAE
  text_encoders/     umt5-xxl (fp8 scaled)
  clip_vision/       OpenCLIP ViT-H
  checkpoints/       SAM3.1 (for the core SAM3_VideoTrack node)

Both Comfy-Org/SCAIL-2 (DiT) and the Wan repackaged encoders are public; SAM3.1 is
GATED (request access at https://huggingface.co/Comfy-Org/sam3.1). HF_TOKEN is
injected from TongFlow Settings (see the Secret.from_dict below).
"""

from __future__ import annotations

import os
from typing import Any

import modal

_cfg: dict[str, Any] = {}

COMFY_MODELS = "/models/comfyui"

# (repo_id, path-in-repo, comfyui-subdir, flat-dest-name)
# Filenames must match workflow.json exactly (the WanAnimatePlus SCAIL-2 graph).
MODELS = [
    (
        "Comfy-Org/SCAIL-2",
        "diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
        "diffusion_models",
        "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
        "loras",
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/vae/wan_2.1_vae.safetensors",
        "vae",
        "wan_2.1_vae.safetensors",
    ),
    (
        "Kijai/WanVideo_comfy",
        "umt5-xxl-enc-fp8_e4m3fn.safetensors",
        "text_encoders",
        "umt5-xxl-enc-fp8_e4m3fn.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/clip_vision/clip_vision_h.safetensors",
        "clip_vision",
        "clip_vision_h.safetensors",
    ),
    # GATED: request access at https://huggingface.co/Comfy-Org/sam3.1
    (
        "Comfy-Org/sam3.1",
        "checkpoints/sam3.1_multiplex_fp16.safetensors",
        "checkpoints",
        "sam3.1_multiplex_fp16.safetensors",
    ),
]

volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)

model_downloader = modal.App("model_downloader")

_download_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.34.0,<1.0"
)


@model_downloader.function(
    image=_download_image,
    volumes={"/models": volume},
    timeout=7200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def _download() -> None:
    import shutil

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError

    token = os.environ.get("HF_TOKEN") or None

    for repo, path, subdir, name in MODELS:
        dest_dir = os.path.join(COMFY_MODELS, subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, name)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"skip (exists): {subdir}/{name}")
            continue
        print(f"Downloading {repo}/{path} ...")
        try:
            src = hf_hub_download(repo_id=repo, filename=path, token=token)
        except GatedRepoError as e:
            raise RuntimeError(
                f"{repo} is gated and this HF_TOKEN is not authorized. Open "
                f"https://huggingface.co/{repo}, accept the license, then re-run."
            ) from e
        shutil.copyfile(src, dest)
        print(f"  got {subdir}/{name} ({os.path.getsize(dest) // (1024 * 1024)} MB)")
        # Commit after each large file so a later failure doesn't re-download it.
        volume.commit()

    print("Done.")


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
