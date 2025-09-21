#!/usr/bin/env python3
import json
from pathlib import Path
from typing import List, Tuple

import torch
from loguru import logger
import tyro

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.text2image import HunyuanDiTPipeline


def _load_tasks(json_path: Path) -> List[Tuple[str, str]]:
    """Load tasks from a JSON file.
    Expected format: {"name": "prompt", ...}.
    Returns a list of (name, prompt) sorted by name.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(
        data, dict), "Input JSON must be a dict mapping name -> prompt"
    items = sorted([(str(k), str(v))
                   for k, v in data.items()], key=lambda x: x[0])
    assert len(items) > 0, "No tasks found in the input JSON"
    return items


def _init_pipelines(
    device: str,
    t2i_repo: str,
    t2mesh_repo: str,
    remove_background: bool,
):
    """Initialize text-to-image and text-to-mesh pipelines on a given device."""
    logger.info(f"[{device}] Loading T2I: {t2i_repo}")
    t2i = HunyuanDiTPipeline(t2i_repo, device=device)

    logger.info(f"[{device}] Loading T2Mesh: {t2mesh_repo}")
    t2mesh = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        t2mesh_repo, device=device)

    rembg = BackgroundRemover() if remove_background else None
    return t2i, t2mesh, rembg


def _process_one(
    name: str,
    prompt: str,
    out_root: Path,
    t2i,
    t2mesh,
    rembg,
) -> None:
    """Generate one mesh from a given prompt and save it to output directory."""
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / f"{name}.glb"

    logger.info(f"Generating image for [{name}]")
    image = t2i(prompt)
    if rembg is not None and getattr(image, "mode", None) == "RGB":
        image = rembg(image)

    logger.info(f"Generating mesh for [{name}]")
    mesh = t2mesh(image=image)[0]
    mesh.export(str(mesh_path))
    logger.success(f"Saved: {mesh_path}")


def _worker(
    gpu_id: int,
    tasks_queue,  # mp.Queue with (name, prompt) items
    out_root: Path,
    t2i_repo: str,
    t2mesh_repo: str,
    remove_background: bool,
    seed: int,
):
    """Worker bound to a single GPU. It repeatedly pulls tasks from the shared queue."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    torch.manual_seed(seed + gpu_id)

    logger.info(f"[GPU {gpu_id}] starting")
    t2i, t2mesh, rembg = _init_pipelines(
        device=device,
        t2i_repo=t2i_repo,
        t2mesh_repo=t2mesh_repo,
        remove_background=remove_background,
    )

    from queue import Empty
    while True:
        try:
            name, prompt = tasks_queue.get_nowait()
        except Empty:
            break

        _process_one(name, prompt, out_root, t2i, t2mesh, rembg)
        torch.cuda.empty_cache()

    logger.success(f"[GPU {gpu_id}] done.")


def main(
    input_json: Path,
    output_dir: Path,
    gpus: List[int] = [0,],
    t2i_repo: str = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
    t2mesh_repo: str = "tencent/Hunyuan3D-2",
    remove_background: bool = True,
    seed: int = 42,
    skip_existing: bool = True,
):
    """
    Multi-GPU text2mesh inference script with dynamic task dispatching.

    Args:
        input_json: Path to JSON mapping name -> prompt
        output_dir: Root dir for results. Mesh -> output_dir/name/name.glb
        gpus: List of GPU ids, e.g. [0,1,2]
        t2i_repo: Text-to-image model repo id
        t2mesh_repo: Text-to-mesh model repo id
        remove_background: Whether to remove background before mesh generation
        seed: Random seed for reproducibility
        skip_existing: Skip tasks whose .glb already exists
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    items = _load_tasks(input_json)
    assert len(gpus) > 0, "No GPUs provided"

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks_queue = ctx.Queue()

    skipped = 0
    for name, prompt in items:
        mesh_path = out_root / name / f"{name}.glb"
        if skip_existing and mesh_path.exists():
            logger.info(f"Skipping [{name}] (already exists: {mesh_path})")
            skipped += 1
            continue
        tasks_queue.put((name, prompt))

    logger.info(
        f"GPUs: {gpus} | total tasks: {len(items)} | queued: {tasks_queue.qsize()} | skipped: {skipped}")

    procs = []
    for gpu_id in gpus:
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, tasks_queue, out_root, t2i_repo,
                  t2mesh_repo, remove_background, seed),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


if __name__ == "__main__":
    tyro.cli(main)
