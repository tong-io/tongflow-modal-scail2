# tongflow-modal-scail2

<img width="1458" height="817" alt="截屏2026-06-17 17 12 01" src="https://github.com/user-attachments/assets/dbe1aae2-49b6-4a86-b5d7-1a1b7ce4d7a2" />

<video src="https://github.com/user-attachments/assets/605c1e49-50cd-4256-9a3c-9a5e659cbe63" />

Official TongFlow plugin. End-to-end controlled character animation with
**SCAIL-2** ([zai-org/SCAIL-2](https://github.com/zai-org/SCAIL-2), Wan2.1-based),
running on a GPU via [Modal](https://modal.com). Same two slots as
`tongflow-modal-wan-animate`, so you can pick either backend.

Runs headless **ComfyUI** (master) + **WanVideoWrapper** + **WanAnimatePlus**
(wuwukaka's SCAIL-2 fork) + the core SAM3 nodes, driving one community-proven
SCAIL-2 graph ([`workflow.json`](workflow.json)). A single boolean in the graph
switches the mode, so both slots share one workflow:

| Node slot                    | Mode        | replacement_mode |
|------------------------------|-------------|------------------|
| `video-image-gen-video-move` | Animation   | `false`          |
| `video-image-gen-video-mix`  | Replacement | `true`           |

Input is a character image + a driving video. `SAM3_VideoTrack` + `SCAIL2ColoredMask`
produce the colored masks inside ComfyUI; output is an mp4 (driving audio preserved).
The **fp8** SCAIL-2 14B DiT with **BlockSwap** keeps it within a single A100-80GB.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `MODAL_TOKEN_ID` | ✅ | Create at [modal.com/settings/tokens](https://modal.com/settings/tokens). |
| `MODAL_TOKEN_SECRET` | ✅ | Paired with `MODAL_TOKEN_ID`. |
| `HF_TOKEN` | ✅ | Must be authorized for the gated `Comfy-Org/sam3.1` (accept its license first). |

### Weights (Hugging Face)

`download.py` fetches ComfyUI-format weights to the shared `models` volume under
`/models/comfyui/<subdir>` (DiT, lightx2v lora, Wan VAE, umt5-xxl, clip vision,
and **SAM3.1**). HF_TOKEN is injected from Settings — no manual `modal secret
create`. **SAM3.1 is gated**: request access at
[Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1) first.

## Usage

```bash
# One-time: fetch weights to the volume
modal run download.py::download

# Deploy the inference app (TongFlow does this automatically on first use)
modal deploy deploy.py
```

The platform invokes `entry.py` per task; it auto-deploys on first use and
re-deploys when `deploy.py` changes. `workflow.json` is mounted at deploy time
(not baked into a cached image layer), so editing it ships on the next deploy.

## Output length & performance

- **Output length = min(driving video length, `duration`).** The animation slot's
  `duration` (seconds) is converted to `frame_load_cap = duration × 24fps`. A 15s
  request only yields 15s if the driving video is at least that long. The
  replacement slot has no `duration` and follows the driving video up to the
  default cap (~5s / 121 frames in `workflow.json` node `#46`).
- **Cost:** ~5s @ 896 long-edge ≈ 9 min on A100-80GB (fp8, 6-step lightx2v,
  BlockSwap 10), including cold-start model load. WanAnimatePlus processes in
  81-frame windows, so wall-time scales roughly linearly with length.

### Tuning knobs

- **`workflow.json` `#51` `blocks_to_swap`** (default 10): lower = faster (more
  layers resident on GPU), higher = safer on VRAM. 0 is fastest on 80GB but may
  OOM on long/high-res clips; raise toward 20 if it does.
- **`workflow.json` `#52` `attention_mode`** (`sdpa`): switch to `sageattn` for a
  speedup, but that requires installing sageattention in `deploy.py`'s image.
- **`deploy.py` `scaledown_window`** (default 5s): raise it to keep the container
  warm between calls and skip cold-start model loading (idle GPU is still billed).
