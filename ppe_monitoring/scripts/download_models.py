from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppe_monitoring.detector import canonical_class


PROJECT_DIR = Path(__file__).resolve().parents[1]
PPE_REPO = "Tanishjain9/yolov8n-ppe-detection-6classes"
HELMET_FALLBACK_REPO = "keremberke/yolov8n-hard-hat-detection"
HELMET_FALLBACK_FILE = "best.pt"


def model_names(model: Any) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def validate_ppe_model(path: Path, *, smoke_test: bool = True) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"PPE model does not exist: {path}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
    except Exception as exc:
        raise RuntimeError(f"PPE checkpoint cannot be loaded by Ultralytics: {path}: {exc}") from exc
    names = model_names(model)
    canonical = {canonical_class(name) for name in names}
    missing = {"helmet", "vest"} - canonical
    if missing:
        raise RuntimeError(
            f"PPE checkpoint {path.name} has classes {names}, missing required aliases: {sorted(missing)}"
        )
    architecture = model.model.__class__.__name__
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    print(f"PPE architecture: {architecture} ({parameters:,} parameters)")
    print(f"PPE model.names: {model.names}")
    if smoke_test:
        try:
            import numpy as np
            sample = np.zeros((640, 640, 3), dtype=np.uint8)
            result = model.predict(sample, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
            detections = 0 if result.boxes is None else len(result.boxes)
            speed = {key: round(float(value), 2) for key, value in result.speed.items()}
            print(f"PPE smoke inference passed: {detections} detections on a blank 640x640 image; speed_ms={speed}")
        except Exception as exc:
            raise RuntimeError(f"PPE checkpoint loaded but inference smoke test failed: {exc}") from exc
    return names


def validate_helmet_fallback_model(path: Path, *, smoke_test: bool = True) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Helmet fallback model does not exist: {path}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(path))
    except Exception as exc:
        raise RuntimeError(f"Helmet fallback checkpoint cannot be loaded: {path}: {exc}") from exc
    names = model_names(model)
    canonical = {canonical_class(name) for name in names}
    if "helmet" not in canonical:
        raise RuntimeError(f"Helmet fallback checkpoint has classes {names}, missing a helmet/Hardhat alias")
    print(f"Helmet fallback model.names: {model.names}")
    if smoke_test:
        try:
            import numpy as np
            sample = np.zeros((640, 640, 3), dtype=np.uint8)
            model.predict(sample, imgsz=640, conf=0.25, device="cpu", verbose=False)
            print("Helmet fallback smoke inference passed")
        except Exception as exc:
            raise RuntimeError(f"Helmet fallback loaded but inference failed: {exc}") from exc
    return names


def _repo_candidates(repo_id: str) -> list[str]:
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc
    api = HfApi()
    files = list_repo_files(repo_id=repo_id, repo_type="model")
    weights = [name for name in files if name.lower().endswith(".pt")]
    if not weights:
        raise RuntimeError(f"Hugging Face repository {repo_id!r} contains no .pt files; inspected {len(files)} files")
    info = api.model_info(repo_id=repo_id, files_metadata=True)
    sizes = {item.rfilename: (item.size if item.size is not None else 10**15) for item in info.siblings}
    # Prefer explicit nano weights, then the smallest checkpoint. Every candidate
    # is still loaded and class-validated before it can become ppe_model.pt.
    return sorted(weights, key=lambda name: (not any(token in name.lower() for token in ("nano", "11n", "v8n")), sizes.get(name, 10**15), name))


def download_ppe(destination: Path, force: bool = False) -> list[str]:
    if destination.exists() and not force:
        names = validate_ppe_model(destination)
        print(f"PPE model already valid: {destination}")
        return names
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc
    candidates = _repo_candidates(PPE_REPO)
    print(f"Inspected mandatory repository {PPE_REPO}; .pt candidates: {candidates}")
    errors = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    for filename in candidates:
        try:
            cached = Path(hf_hub_download(repo_id=PPE_REPO, filename=filename, repo_type="model"))
            names = validate_ppe_model(cached)
            temporary = destination.with_suffix(".pt.part")
            shutil.copy2(cached, temporary)
            temporary.replace(destination)
            # Validate the final file, not only the Hugging Face cache entry.
            names = validate_ppe_model(destination)
            print(f"PPE model saved: {destination}")
            print(f"Detected PPE classes: {names}")
            return names
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
    raise RuntimeError("No compatible Ultralytics PPE checkpoint found:\n- " + "\n- ".join(errors))


def download_helmet_fallback(destination: Path, force: bool = False) -> list[str]:
    if destination.exists() and not force:
        names = validate_helmet_fallback_model(destination)
        print(f"Helmet fallback model already valid: {destination}")
        return names
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is not installed") from exc
    cached = Path(hf_hub_download(
        repo_id=HELMET_FALLBACK_REPO,
        filename=HELMET_FALLBACK_FILE,
        repo_type="model",
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.part")
    shutil.copy2(cached, temporary)
    temporary.replace(destination)
    names = validate_helmet_fallback_model(destination)
    print(f"Helmet fallback saved: {destination} from {HELMET_FALLBACK_REPO}/{HELMET_FALLBACK_FILE}")
    return names


def download_pose(destination: Path, force: bool = False) -> list[str]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed") from exc
    if destination.exists() and not force:
        try:
            model = YOLO(str(destination))
            if getattr(model.model, "kpt_shape", None) is None:
                raise RuntimeError("checkpoint has no pose keypoints")
            print(f"Pose model already valid: {destination}")
            return model_names(model)
        except Exception as exc:
            raise RuntimeError(f"Existing pose model is invalid: {destination}: {exc}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for official_name in ("yolo11n-pose.pt", "yolov8n-pose.pt"):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                previous_directory = Path.cwd()
                try:
                    os.chdir(temp_dir)
                    model = YOLO(official_name)
                    source = Path(str(model.ckpt_path)).resolve()
                finally:
                    os.chdir(previous_directory)
                if not source.is_file() or getattr(model.model, "kpt_shape", None) is None:
                    raise RuntimeError("downloaded checkpoint is not a pose model")
                temporary = destination.with_suffix(".pt.part")
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            verified = YOLO(str(destination))
            names = model_names(verified)
            print(f"Pose model saved: {destination} (source {official_name})")
            return names
        except Exception as exc:
            errors.append(f"{official_name}: {exc}")
    raise RuntimeError("Unable to download a compatible nano pose model:\n- " + "\n- ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify primary PPE, helmet fallback, and pose models")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--force", action="store_true", help="Re-download and replace valid models")
    parser.add_argument("--verify-only", action="store_true", help="Do not download; validate existing files")
    args = parser.parse_args()
    ppe_path = args.models_dir / "ppe_model.pt"
    helmet_fallback_path = args.models_dir / "helmet_fallback.pt"
    pose_path = args.models_dir / "pose_model.pt"
    try:
        if args.verify_only:
            print(f"Detected PPE classes: {validate_ppe_model(ppe_path)}")
            print(f"Detected helmet fallback classes: {validate_helmet_fallback_model(helmet_fallback_path)}")
            from ultralytics import YOLO
            pose = YOLO(str(pose_path))
            if getattr(pose.model, "kpt_shape", None) is None:
                raise RuntimeError(f"Pose model has no keypoints: {pose_path}")
            print(f"Pose model valid: {pose_path}")
        else:
            download_ppe(ppe_path, args.force)
            download_helmet_fallback(helmet_fallback_path, args.force)
            download_pose(pose_path, args.force)
        if not ppe_path.is_file() or not helmet_fallback_path.is_file() or not pose_path.is_file():
            raise RuntimeError("Model setup did not produce all three required model files")
        return 0
    except Exception as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
