"""Modal deploy entry for SCAIL-2 (ComfyUI / WanAnimatePlus path).

Implements both video slots via one community-proven WanAnimatePlus SCAIL-2 graph
(workflow.json). A single boolean (node #188) switches the mode:

- ``video-image-gen-video-move`` → animation   (replacement_mode=False)
- ``video-image-gen-video-mix``  → replacement  (replacement_mode=True)

Runs headless ComfyUI master (for the core SAM3_VideoTrack / SCAIL2ColoredMask
nodes) + WanVideoWrapper + WanAnimatePlus + helper node packs. The fp8 SCAIL-2 14B
DiT with BlockSwap (20 blocks offloaded) fits a single A100-80GB. The server boots
once (@modal.enter) and is reused across calls.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import modal
from tongflow import deploy
from tongflow.models.video_image_gen_video_mix import (
    VideoImageGenVideoMixInput,
    VideoImageGenVideoMixOutput,
)
from tongflow.models.video_image_gen_video_move import (
    VideoImageGenVideoMoveInput,
    VideoImageGenVideoMoveOutput,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

COMFY = "/opt/ComfyUI"
COMFY_MODELS = "/models/comfyui"
WORKFLOW_PATH = "/opt/workflow.json"
COMFY_LOG = "/tmp/comfy.log"

volume = modal.Volume.from_name("models", create_if_missing=True)

CUSTOM_NODES = {
    # WanAnimatePlus extends WanVideoWrapper; both are required for the SCAIL_2 nodes.
    "ComfyUI-WanVideoWrapper": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
    "ComfyUI-WanAnimatePlus": "https://github.com/wuwukaka/ComfyUI-WanAnimatePlus.git",
    "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    "ComfyUI_LayerStyle": "https://github.com/chflame163/ComfyUI_LayerStyle.git",
    "ComfyUI-Crystools": "https://github.com/crystian/ComfyUI-Crystools.git",
    "ComfyUI-Easy-Use": "https://github.com/yolain/ComfyUI-Easy-Use.git",
}
_clone_cmds = []
for _name, _url in CUSTOM_NODES.items():
    _dst = f"{COMFY}/custom_nodes/{_name}"
    _clone_cmds.append(f"git clone --depth 1 {_url} {_dst}")
    _clone_cmds.append(
        f"[ -f {_dst}/requirements.txt ] && pip install -r {_dst}/requirements.txt || true"
    )

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git " + COMFY,
        f"pip install -r {COMFY}/requirements.txt",
        *_clone_cmds,
    )
    .pip_install("tongflow==0.1.0")
    .env({"PYTHONPATH": COMFY, "HF_HOME": "/models/hf"})
    # Mounted at runtime (copy defaults to False) so every deploy ships the latest
    # workflow.json. copy=True bakes it into an image layer that can cache stale.
    .add_local_file(
        str(Path(__file__).resolve().parent / "workflow.json"),
        WORKFLOW_PATH,
    )
)

with image.imports():
    import json
    import subprocess
    import time
    import urllib.error
    import urllib.request


def _tail_log(n: int = 3500) -> str:
    """Last n chars of the ComfyUI server stdout (per-node execution trace),
    with download progress-bar spam stripped so the node trace stays visible."""
    try:
        with open(COMFY_LOG, "rb") as f:
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return "(no server log)"
    lines = [
        ln.strip()
        for ln in text.replace("\r", "\n").split("\n")
        if ln.strip() and "MB/s" not in ln and "B/s]" not in ln
    ]
    return "\n".join(lines)[-n:]


def _maybe_bytes(val: object) -> Optional[bytes]:
    if val is None:
        return None
    try:
        return prompt_media_to_bytes(val)
    except (TypeError, ValueError):
        return None


def _aligned_long_side(width: object, height: object, default: int = 896) -> int:
    """Longest-side target (multiple of 32) for the ImageScaleByAspectRatio node."""
    vals = []
    for v in (width, height):
        try:
            n = int(v) if v is not None else 0
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            vals.append(n)
    longest = max(vals) if vals else default
    return max(32, round(longest / 32) * 32)


def _submit_graph(base, wf):
    """Submit an API workflow, poll, return (True, mp4_bytes) or (False, error)."""
    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(
        f"{base}/prompt", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        return False, f"workflow rejected: {e.read().decode()[:1500]}"
    out = None
    final_status = {}
    for _ in range(3600):
        time.sleep(1)
        with urllib.request.urlopen(f"{base}/history/{pid}", timeout=10) as r:
            hist = json.loads(r.read())
        if pid not in hist:
            continue
        h = hist[pid]
        status = h.get("status", {})
        final_status = status
        if status.get("status_str") == "error":
            # Dump the execution_error messages so the failing node is visible.
            return False, (
                "comfy error: " + json.dumps(status.get("messages", status))[:1500]
                + "\n[server log]\n" + _tail_log()
            )
        if h.get("outputs") and status.get("completed"):
            out = h["outputs"]
            break
    if not out:
        return False, "timed out\n[server log]\n" + _tail_log()
    for node_out in out.values():
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key, []):
                fn, sub = item.get("filename"), item.get("subfolder", "")
                typ = item.get("type", "output")
                d = {"output": "output", "temp": "temp"}.get(typ, "output")
                path = os.path.join(COMFY, d, sub, fn or "")
                if fn and fn.endswith((".mp4", ".webm")) and os.path.isfile(path):
                    with open(path, "rb") as fh:
                        raw = fh.read()
                    if raw:
                        return True, raw
    # Diagnostic: show what each node produced + status messages (execution_error
    # lives here even when status_str isn't "error").
    summary = {}
    for nid, node_out in out.items():
        keys = {k: node_out.get(k) for k in ("gifs", "videos", "images") if node_out.get(k)}
        if keys:
            summary[nid] = keys
    return False, (
        "no video output; outputs=" + json.dumps(summary)[:600]
        + "; messages=" + json.dumps(final_status.get("messages", final_status))[:600]
        + "\n[server log]\n" + _tail_log()
    )


@deploy
@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=5,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        """Boot the ComfyUI server once; reused across calls (models stay warm)."""
        os.makedirs(COMFY_MODELS, exist_ok=True)
        with open(os.path.join(COMFY, "extra_model_paths.yaml"), "w") as f:
            f.write(
                "scail_volume:\n"
                f"  base_path: {COMFY_MODELS}/\n"
                "  diffusion_models: diffusion_models\n"
                "  vae: vae\n"
                "  text_encoders: text_encoders\n"
                "  clip_vision: clip_vision\n"
                "  loras: loras\n"
                "  checkpoints: checkpoints\n"
            )
        with open(WORKFLOW_PATH, encoding="utf-8") as fh:
            self._wf_template = fh.read()
        self._logfh = open(COMFY_LOG, "wb")
        self.proc = subprocess.Popen(
            [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-auto-launch",
            ],
            cwd=COMFY,
            stdout=self._logfh,
            stderr=subprocess.STDOUT,
        )
        self.base = "http://127.0.0.1:8188"
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"ComfyUI exited early: {self.proc.returncode}")
            try:
                with urllib.request.urlopen(f"{self.base}/object_info", timeout=2) as r:
                    if r.status == 200:
                        json.loads(r.read())
                        return
            except Exception:
                time.sleep(1)
        raise RuntimeError("ComfyUI server did not become ready")

    @modal.exit()
    def _shutdown(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _run(self, img_b, vid_b, text, seed, width, height, duration, replacement_mode):
        os.makedirs(f"{COMFY}/input", exist_ok=True)
        with open(f"{COMFY}/input/ref.png", "wb") as f:
            f.write(img_b)
        with open(f"{COMFY}/input/driving.mp4", "wb") as f:
            f.write(vid_b)

        wf = json.loads(self._wf_template)
        wf["47"]["inputs"]["image"] = "ref.png"
        wf["46"]["inputs"]["video"] = "driving.mp4"
        wf["56"]["inputs"]["positive_prompt"] = (text or "").strip()
        wf["42"]["inputs"]["seed"] = seed
        # replacement_mode is consumed by the embeds + colored-mask nodes directly.
        wf["57"]["inputs"]["replacement_mode"] = bool(replacement_mode)
        wf["96"]["inputs"]["replacement_mode"] = bool(replacement_mode)
        # Longest-side target feeds both ImageScaleByAspectRatio nodes (driving + ref).
        ws = _aligned_long_side(width, height)
        wf["48"]["inputs"]["scale_to_length"] = ws
        wf["49"]["inputs"]["scale_to_length"] = ws
        # duration (seconds) caps how many driving frames load: frames = duration * fps.
        # Output length = min(driving video length, this cap).
        if duration:
            try:
                wf["46"]["inputs"]["frame_load_cap"] = max(1, round(float(duration) * 24))
            except (TypeError, ValueError):
                pass
        return _submit_graph(self.base, wf)

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MOVE)
    def video_image_gen_video_move(
        self, input: VideoImageGenVideoMoveInput
    ) -> VideoImageGenVideoMoveOutput:
        """Animation: reenact the driving motion onto the character image."""
        img_b = _maybe_bytes(input.image)
        if not img_b:
            return VideoImageGenVideoMoveOutput(success=False, error="Missing image")
        vid_b = _maybe_bytes(input.video)
        if not vid_b:
            return VideoImageGenVideoMoveOutput(
                success=False, error="Missing driving video"
            )
        seed = int(input.seed) if input.seed is not None else 42
        ok, res = self._run(
            img_b, vid_b, input.text, seed, input.width, input.height, input.duration, False
        )
        if ok:
            return VideoImageGenVideoMoveOutput(
                success=True, video=asset(res, mime="video/mp4")
            )
        return VideoImageGenVideoMoveOutput(success=False, error=str(res))

    @modal.method()
    @node_slot(NodeSlots.VIDEO_IMAGE_GEN_VIDEO_MIX)
    def video_image_gen_video_mix(
        self, input: VideoImageGenVideoMixInput
    ) -> VideoImageGenVideoMixOutput:
        """Replacement: swap the person in the driving video with the character."""
        img_b = _maybe_bytes(input.image)
        if not img_b:
            return VideoImageGenVideoMixOutput(success=False, error="Missing image")
        vid_b = _maybe_bytes(input.video)
        if not vid_b:
            return VideoImageGenVideoMixOutput(
                success=False, error="Missing driving video"
            )
        ok, res = self._run(img_b, vid_b, input.text, 42, None, None, None, True)
        if ok:
            return VideoImageGenVideoMixOutput(
                success=True, video=asset(res, mime="video/mp4")
            )
        return VideoImageGenVideoMixOutput(success=False, error=str(res))
