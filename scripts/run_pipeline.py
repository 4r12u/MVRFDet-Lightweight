#!/usr/bin/env python3
"""Run selected MVRFDet stages over unpacked packages or local source archives."""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvrfdet.core.types import MultiViewFeatures
from mvrfdet.offline.sensitive_api.deepseek import DeepSeekVerificationError
from mvrfdet.offline.sensitive_api.package_reasoning import (
    PackageReasoningOptions,
    add_package_reasoning_arguments,
    options_from_args,
    run_package_reasoning,
)
from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.reporting import load_classifier, write_feature_matrix, write_inference, write_package_report
from mvrfdet.stage1.suspicious_api.locator import load_sensitive_api_list
from mvrfdet.stage3.model import LightweightMVRFDet

from smoke_dataset import _safe_source_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SUFFIX = ".tar.gz"
STAGE_INPUTS = {"stage2": ("stage1", "features.npz"), "stage3": ("stage2", "matrix.npy")}
LEGACY_MODES = {
    "all": "Run all stages and DeepSeek reasoning",
    "lite": "Run without LLM reasoning, with optional prediction",
    "stage1": "Extract features and graphs from packages",
    "stage2": "Extract features and build matrices from packages",
    "stage3": "Run stages 1–3 from packages with optional prediction",
    "showfig": "Generate API call graph and CFG visuals",
}


@dataclass(frozen=True)
class DatasetInput:
    path: Path
    name: str
    report_key: Path
    is_archive: bool


def _is_package_dir(path: Path) -> bool:
    try:
        metadata_files = list(path.rglob("PKG-INFO")) + list(path.rglob("METADATA"))
    except OSError:
        return False
    return bool(metadata_files and any(path.rglob("*.py")))


def _is_package_root(path: Path) -> bool:
    if any((path / name).is_file() for name in ("PKG-INFO", "METADATA")):
        return _is_package_dir(path)
    # Source distributions often unpack into a same-named inner directory.
    matching_children = [
        child for child in path.iterdir()
        if child.is_dir() and child.name.casefold() == path.name.casefold() and _is_package_dir(child)
    ]
    return len(matching_children) == 1


def discover_inputs(dataset: Path) -> list[DatasetInput]:
    """Find archives and unpacked packages without treating dataset containers as packages."""
    root = dataset.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if root.is_file():
        if not root.name.endswith(ARCHIVE_SUFFIX):
            raise ValueError(f"Expected a .tar.gz source archive: {root}")
        return [DatasetInput(root, root.name, Path(root.name[:-7]), True)]

    found: dict[str, DatasetInput] = {}

    def visit(folder: Path) -> None:
        try:
            children = sorted(folder.iterdir())
        except OSError:
            return
        direct_archives = [item for item in children if item.is_file() and item.name.endswith(ARCHIVE_SUFFIX)]
        archive_stems = {item.name[:-7] for item in direct_archives}
        for archive in direct_archives:
            relative = archive.relative_to(root)
            report_key = relative.parent / archive.name[:-7]
            found[str(archive)] = DatasetInput(archive, archive.name, report_key, True)

        child_dirs = [item for item in children if item.is_dir()]
        direct_metadata = any((folder / name).is_file() for name in ("PKG-INFO", "METADATA"))
        if direct_metadata and any(folder.rglob("*.py")):
            if not direct_archives:
                relative = folder.relative_to(root) if folder != root else Path(folder.name)
                found[str(folder)] = DatasetInput(folder, folder.name, relative, False)
            return

        for child in child_dirs:
            if child.name in archive_stems:
                continue
            try:
                has_nested_archive = next(child.rglob("*.tar.gz"), None) is not None
            except OSError:
                has_nested_archive = False
            if has_nested_archive:
                visit(child)
            elif _is_package_root(child):
                relative = child.relative_to(root)
                found[str(child)] = DatasetInput(child, child.name, relative, False)
            else:
                visit(child)

    visit(root)
    if not found and _is_package_dir(root):
        found[str(root)] = DatasetInput(root, root.name, Path(root.name), False)
    return sorted(found.values(), key=lambda item: (str(item.report_key).casefold(), item.name.casefold()))


def discover_stage_inputs(source: Path, previous_stage: str, filename: str) -> list[DatasetInput]:
    root = source.resolve(strict=True)
    if root.is_file():
        if root.suffix.casefold() != Path(filename).suffix:
            raise ValueError(f"Expected a {Path(filename).suffix} file: {root}")
        files = [root]
    else:
        files = sorted(path for path in root.rglob(filename)
                       if path.is_file() and path.resolve().is_relative_to(root))

    items = []
    keys = set()
    for path in files:
        if path.name == filename and path.parent.name == previous_stage:
            package_dir = path.parent.parent
            key = (package_dir.relative_to(root) if package_dir != root and package_dir.is_relative_to(root)
                   else Path(package_dir.name))
        else:
            key = path.relative_to(root).with_suffix("") if root.is_dir() else Path(path.stem)
        if key in keys:
            raise ValueError(f"Multiple {filename} files map to package {key}")
        keys.add(key)
        items.append(DatasetInput(path, key.name, key, False))
    return items


def _stages_for_mode(mode: str) -> set[str]:
    return {
        "all": {"stage1", "stage2", "stage3"},
        "lite": {"stage1", "stage2", "stage3"},
        "full": {"stage1", "stage2", "stage3"},
        "stage1": {"stage1"},
        "stage2": {"stage1", "stage2"},
        "stage3": {"stage1", "stage2", "stage3"},
        "showfig": {"stage1"},
    }[mode]


def _run_input(
    item: DatasetInput,
    sensitive_list: Path,
    report_dir: Path,
    backend: str,
    checkpoint: Path | None,
    stages: set[str],
    llm_options: PackageReasoningOptions | None,
    classifier: LightweightMVRFDet | None = None,
) -> dict[str, object]:
    if item.is_archive:
        with tempfile.TemporaryDirectory(prefix="mvrfdet-run-") as temporary:
            source_root = Path(temporary)
            return _run_package(
                item, source_root, sensitive_list, report_dir, backend,
                checkpoint, stages, llm_options, classifier,
            )
    return _run_package(
        item, item.path, sensitive_list, report_dir, backend,
        checkpoint, stages, llm_options, classifier,
    )


def _run_package(
    item: DatasetInput,
    source_root: Path,
    sensitive_list: Path,
    report_dir: Path,
    backend: str,
    checkpoint: Path | None,
    stages: set[str],
    llm_options: PackageReasoningOptions | None,
    classifier: LightweightMVRFDet | None,
) -> dict[str, object]:
    if item.is_archive:
        _safe_source_files(item.path, source_root)
    sensitive = load_sensitive_api_list(sensitive_list)
    artifacts = LightweightFeaturePipeline(static_backend=backend).extract_with_artifacts(source_root, sensitive)
    result = write_package_report(
        report_dir, item.name, source_root, artifacts,
        checkpoint=checkpoint, stages=stages, classifier=classifier,
    )
    if llm_options is not None:
        result.update(run_package_reasoning(artifacts, sensitive, report_dir, llm_options))
    return result


def _write_run_summary(output_dir: Path, mode: str, results: list[dict[str, object]], dataset: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary = {
        "mode": mode,
        "dataset": str(dataset.resolve()),
        "packages": len(results),
        "completed": sum(item.get("status") == "completed" for item in results),
        "errors": sum(item.get("status") == "error" for item in results),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Run stage1 with packages, stage2 with saved features, stage3 with saved matrices, "
        "or full with packages to run stages 1–3 in one pass."
    ))
    parser.add_argument("command", choices=("stage1", "stage2", "stage3", "full"), nargs="?")
    parser.add_argument("input", type=Path, nargs="?", metavar="INPUT",
                        help="Package/archive or output from the previous stage")
    modes = parser.add_mutually_exclusive_group()
    for mode, label in LEGACY_MODES.items():
        modes.add_argument(f"--{mode}", dest=f"{mode}_dataset", nargs="?", const="",
                           metavar="DATASET", help=f"Legacy: {label}")

    parser.add_argument("--sensitive-list", type=Path, default=Path("data/sensitive_api/sensitive_apis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/pipeline_reports"))
    parser.add_argument("--checkpoint", type=Path, help="Trained checkpoint required by stage3 and full")
    parser.add_argument("--backend", choices=["auto", "ast", "scalpel"], default="ast")
    parser.add_argument("--only", help="Filter package/archive names with a glob")
    parser.add_argument("--limit", type=int, help="Maximum number of packages to process")
    add_package_reasoning_arguments(parser, include_enable_flag=False)
    args = parser.parse_args()

    legacy = next(((mode, getattr(args, f"{mode}_dataset")) for mode in LEGACY_MODES
                   if getattr(args, f"{mode}_dataset") is not None), None)
    if args.command is not None:
        if legacy is not None:
            parser.error("Use a stage command or a legacy mode flag separately")
        mode, input_path = args.command, args.input
    elif legacy is not None:
        mode, dataset_arg = legacy
        input_path = Path(dataset_arg) if dataset_arg else None
    else:
        parser.error("Choose stage1, stage2, stage3, or full")
    stage_input = STAGE_INPUTS.get(mode) if args.command else None
    if stage_input and input_path is None:
        parser.error(f"{mode} requires a {stage_input[1]} file or the {stage_input[0]} output directory")
    dataset = input_path.expanduser() if input_path is not None else PROJECT_ROOT / "dataset"

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if stage_input is None and not args.sensitive_list.is_file():
        parser.error(f"Sensitive API list not found: {args.sensitive_list}")
    if args.command in {"stage3", "full"} and args.checkpoint is None:
        parser.error(f"{mode} requires --checkpoint")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    llm_options = None
    if mode == "all":
        try:
            llm_options = options_from_args(args)
        except (ValueError, OSError, DeepSeekVerificationError) as exc:
            parser.error(str(exc))

    try:
        inputs = discover_stage_inputs(dataset, *stage_input) if stage_input else discover_inputs(dataset)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.only:
        inputs = [item for item in inputs if fnmatch.fnmatch(item.name, args.only)]
    if args.limit is not None:
        inputs = inputs[:args.limit]
    if not inputs:
        expected = stage_input[1] if stage_input else "package directories or .tar.gz archives"
        parser.error(f"No matching {expected} found under {dataset}")

    model = None
    if args.command in {"stage3", "full"}:
        try:
            model = load_classifier(args.checkpoint)
        except (OSError, ValueError, RuntimeError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            parser.error(f"Cannot load checkpoint: {exc}")

    mode_output = args.output_dir / mode
    mode_output.mkdir(parents=True, exist_ok=True)
    stages = _stages_for_mode(mode)
    results = []
    for index, item in enumerate(inputs, 1):
        report_dir = mode_output / item.report_key
        started = time.monotonic()
        try:
            if stage_input and mode == "stage2":
                write_feature_matrix(report_dir / mode, MultiViewFeatures.load_npz(item.path))
                result = {"artifact": str(report_dir / mode / "matrix.npy")}
            elif stage_input and mode == "stage3":
                with item.path.open("rb") as handle:
                    matrix = np.load(handle, allow_pickle=False)
                    write_inference(report_dir / mode, matrix, args.checkpoint, model)
                result = {"artifact": str(report_dir / mode / "inference.json"), "stage3_status": "completed"}
            else:
                result = _run_input(
                    item, args.sensitive_list, report_dir, args.backend,
                    args.checkpoint, stages, llm_options, model,
                )
            result = {"package": item.name, "status": "completed", **result}
        except Exception as exc:
            report_dir.mkdir(parents=True, exist_ok=True)
            error_text = f"{type(exc).__name__}: {exc}"
            (report_dir / "summary.json").write_text(json.dumps({
                "package": item.name, "status": "error", "error": error_text,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            (report_dir / "report.html").write_text(
                "<!doctype html><html><head><meta charset='utf-8'></head><body>"
                f"<h1>{html.escape(item.name)}</h1><p>Status: error</p>"
                f"<pre>{html.escape(error_text)}</pre></body></html>",
                encoding="utf-8",
            )
            result = {"package": item.name, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "report": str(report_dir / "report.html")}
        result["seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        print(f"{index:03d}/{len(inputs):03d} {item.name}: {result['status']} ({result['seconds']}s)", flush=True)

    summary_path = _write_run_summary(mode_output, mode, results, dataset)
    print(f"Summary: {summary_path}")
    print(f"Reports: {mode_output}")
    if args.command and any(result["status"] == "error" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
