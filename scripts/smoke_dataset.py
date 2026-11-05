"""Run static feature extraction over local source archives without executing samples."""

from __future__ import annotations

import argparse
import fnmatch
import html
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.offline.sensitive_api.deepseek import DeepSeekVerificationError
from mvrfdet.offline.sensitive_api.package_reasoning import (
    PackageReasoningOptions,
    add_package_reasoning_arguments,
    options_from_args,
    run_package_reasoning,
)
from mvrfdet.reporting import write_package_report
from mvrfdet.stage1.suspicious_api.locator import load_sensitive_api_list


def _safe_source_files(archive: Path, destination: Path) -> None:
    total_bytes = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                continue
            if any(":" in part or "\\" in part for part in name.parts):
                continue
            if name.suffix != ".py" and name.name not in {"PKG-INFO", "METADATA"}:
                continue
            if member.size > 20_000_000 or total_bytes + member.size > 40_000_000:
                raise ValueError("Source files exceed the smoke test extraction limit")
            source = handle.extractfile(member)
            if source is None:
                continue
            content = source.read(member.size + 1)
            if len(content) != member.size:
                raise ValueError(f"Incomplete archive member: {member.name}")
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            total_bytes += len(content)


def _worker(
    archive: Path, sensitive_list: Path, report_dir: Path,
    checkpoint: Path | None, llm_options: PackageReasoningOptions | None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="mvrfdet-smoke-") as temporary:
        root = Path(temporary)
        _safe_source_files(archive, root)
        sensitive = load_sensitive_api_list(sensitive_list)
        artifacts = LightweightFeaturePipeline(static_backend="ast").extract_with_artifacts(root, sensitive)
        result = write_package_report(report_dir, archive.name, root, artifacts, checkpoint)
        if llm_options is not None:
            result.update(run_package_reasoning(artifacts, sensitive, report_dir, llm_options))
        print(json.dumps(result, ensure_ascii=False))


def write_failure_report(report_dir: Path, result: dict) -> None:
    """Give failed or timed-out packages a visible per-package status page."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    package = html.escape(str(result["package"]))
    status = html.escape(str(result["status"]))
    error = html.escape(str(result.get("error", "")))
    (report_dir / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>{package} · {status}</title></head>"
        f"<body><h1>{package}</h1><p>Analysis status: <b>{status}</b></p><pre>{error}</pre></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--sensitive-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, help="Whole-package timeout in seconds; computed from LLM settings when enabled")
    parser.add_argument("--only", help="Archive filename glob, for example 1inch-8.7.tar.gz")
    parser.add_argument("--limit", type=int, help="Maximum number of selected archives")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--package-report-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    add_package_reasoning_arguments(parser)
    args = parser.parse_args()
    if args.timeout is not None and args.timeout < 1:
        parser.error("--timeout must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        llm_options = options_from_args(args) if args.llm_reasoning else None
    except (ValueError, OSError, DeepSeekVerificationError) as exc:
        parser.error(str(exc))

    if args.worker:
        if args.package_report_dir is None:
            parser.error("--package-report-dir is required for --worker")
        _worker(args.worker, args.sensitive_list, args.package_report_dir, args.checkpoint, llm_options)
        return
    if args.dataset_dir is None:
        parser.error("--dataset-dir is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timeout = args.timeout or (
        30 + llm_options.max_apis * (
            llm_options.retries * llm_options.timeout + max(0, 2 ** (llm_options.retries - 1) - 1)
        ) if llm_options else 30
    )
    archives = sorted(args.dataset_dir.glob("*.tar.gz"))
    if args.only:
        archives = [archive for archive in archives if fnmatch.fnmatch(archive.name, args.only)]
    if args.limit is not None:
        archives = archives[:args.limit]
    if not archives:
        parser.error("No .tar.gz archives matched the selected dataset and filters")
    results = []
    for archive in archives:
        report_dir = args.output_dir / archive.name[:-7]
        started = time.monotonic()
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker", str(archive.resolve()),
            "--sensitive-list", str(args.sensitive_list.resolve()),
            "--package-report-dir", str(report_dir.resolve()),
            "--output-dir", str(args.output_dir.resolve()),
        ]
        if args.checkpoint is not None:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        if llm_options is not None:
            command.extend([
                "--llm-reasoning", "--llm-api-key-env", llm_options.key_env,
                "--llm-model", llm_options.model,
                "--llm-max-apis", str(llm_options.max_apis),
                "--llm-contexts-per-api", str(llm_options.contexts_per_api),
                "--llm-prompt", str(llm_options.prompt_path.resolve()),
                "--llm-timeout", str(llm_options.timeout),
                "--llm-retries", str(llm_options.retries),
                "--llm-max-tokens", str(llm_options.max_tokens),
            ])
            if llm_options.api_url is not None:
                command.extend(["--llm-api-url", llm_options.api_url])
            if llm_options.api_name is not None:
                command.extend(["--llm-api", llm_options.api_name])
            if llm_options.refresh:
                command.append("--llm-refresh")
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            if run.returncode == 0:
                result = {"package": archive.name, "status": "ok", **json.loads(run.stdout)}
            else:
                result = {"package": archive.name, "status": "error", "error": run.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            result = {"package": archive.name, "status": "timeout", "error": f"> {timeout}s"}
        result["seconds"] = round(time.monotonic() - started, 2)
        if result["status"] != "ok":
            write_failure_report(report_dir, result)
        results.append(result)
        print(f"{len(results):02d} {archive.name}: {result['status']} ({result['seconds']}s)", flush=True)

    summary = {
        "backend": "ast",
        "package_timeout": timeout,
        "sensitive_list": str(args.sensitive_list.resolve()),
        "packages": len(results),
        "ok": sum(item["status"] == "ok" for item in results),
        "errors": sum(item["status"] == "error" for item in results),
        "timeouts": sum(item["status"] == "timeout" for item in results),
        "results": results,
    }
    report = args.output_dir / ("summary.selection.json" if args.only or args.limit else "summary.json")
    if args.only or args.limit:
        summary["selection"] = {"only": args.only, "limit": args.limit}
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {report}", flush=True)


if __name__ == "__main__":
    main()
