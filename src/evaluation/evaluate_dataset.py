"""
evaluate_dataset.py
====================
Evaluation Framework — main CLI entry point and workflow coordinator.

This module's only job is orchestration: parse arguments, resolve the
dataset input (folder or ZIP), run the existing QR Shield pipeline over
every image in parallel, and hand the results to the specialised modules
that do the actual work — ``dataset_loader`` (discovery/ZIP/cleanup),
``duplicate_detector`` (skip re-processing identical images),
``checkpoint`` (resume support), ``progress`` (console feedback),
``metrics``/``benchmark`` (all numbers), ``plots``/``generate_report``/
``html_report`` (all presentation), ``gallery`` (image galleries), and
``system_info`` (environment snapshot). No metric computation, chart
rendering, or report formatting logic lives in this file.

Pipeline reuse
--------------
The actual QR Shield processing steps are reused as-is, without
modification, from ``src.integration.main``, ``src.preprocessing
.image_enhancement``, and ``src.risk_assessment.risk_engine`` — see
:func:`process_single_image`'s docstring for exactly which functions are
called and why ``main.py``'s own ``run_pipeline()`` isn't used verbatim.

Output layout
-------------
::

    results/<dataset_name>/
        csv/          results.csv, benchmark.csv, category_summary.csv, ...
        json/         results.json, summary.json, benchmark.json
        reports/      Evaluation_Report.md, Evaluation_Report.html
        charts/       every generated PNG/SVG/PDF chart
        gallery/      detected/, failed/, high_risk/, tampered/
        failed_images/  every failed image, uncapped
        logs/         evaluation.log
        checkpoints/  resume.json

Usage
-----
::

    python -m src.evaluation.evaluate_dataset data/evaluation \\
        --recursive --save-json --save-csv --generate-plots --generate-report \\
        --workers 4 --verbose

    python -m src.evaluation.evaluate_dataset datasets/batch1.zip --resume

Note on ground truth
--------------------
This evaluation dataset has no per-image label files; ground truth is
derived from the category folder name via
``src.evaluation.utils.load_category_labels`` (overridable with
``--labels labels.json``). See that module's docstring for the exact
convention used.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.evaluation import config
from src.evaluation.benchmark import build_benchmark_report
from src.evaluation.checkpoint import Checkpoint, load_checkpoint, save_checkpoint, should_checkpoint
from src.evaluation.dataset_loader import discover_images, resolve_dataset_source
from src.evaluation.duplicate_detector import clone_result_for_duplicates, find_duplicates
from src.evaluation.gallery import build_galleries, copy_failed_images
from src.evaluation.generate_report import generate_markdown_report, write_report
from src.evaluation.html_report import generate_html_report, write_html_report
from src.evaluation.metrics import build_confusion_matrix, compute_binary_metrics, compute_numeric_stats
from src.evaluation.progress import ProgressReporter
from src.evaluation.system_info import collect_system_info
from src.evaluation.url_analyzer_adapter import run_url_analysis
from src.evaluation.utils import (
    StageTimer,
    format_duration,
    ground_truth_for,
    load_category_labels,
    setup_logging,
    write_csv,
    write_json,
)

logger = logging.getLogger("evaluation.evaluate_dataset")


# ===========================================================================
# Per-image pipeline execution (reuses main.py / risk_engine.py as-is)
# ===========================================================================

def process_single_image(
    image_path: str,
    category: str,
    ground_truth: dict[str, bool],
    risk_severity_threshold: int,
) -> dict[str, Any]:
    """Run the full existing QR Shield pipeline on one image and score it.

    Reuses, without modification:

    * ``src.integration.main.step_load_image``
    * ``src.integration.main.step_preprocess``
    * ``src.integration.main.step_detect_qr``
    * ``src.preprocessing.image_enhancement.remap_to_original``
    * ``src.risk_assessment.risk_engine.RiskEngine.assess``

    Note: ``src.integration.main.run_pipeline`` itself is a CLI-oriented,
    print-and-exit-code function with no risk assessment step, so it is not
    reusable verbatim for bulk, in-process, risk-scored evaluation; this
    function instead re-composes the same step functions main.py already
    exposes, in the same order main.py uses them, and adds the risk
    assessment call that main.py does not yet wire in. No detection,
    preprocessing, or scoring logic is reimplemented here.

    Parameters
    ----------
    image_path : str
        Absolute path to the image file.
    category : str
        Dataset category (subfolder name) the image belongs to.
    ground_truth : dict[str, bool]
        ``{"expect_qr": bool, "expect_malicious": bool}`` for this category.
    risk_severity_threshold : int
        Minimum ``RiskLevel.severity`` counted as "predicted malicious".

    Returns
    -------
    dict[str, Any]
        Flat result record; always returned, never raises. On failure,
        ``success`` is False and ``error`` holds a description.
    """
    # Imports are local to this function so that each worker **process**
    # (spawned by ProcessPoolExecutor) performs its own module import and
    # engine construction, avoiding any attempt to pickle live cv2/engine
    # objects across the process boundary.
    import cv2

    from src.integration.main import step_detect_qr, step_load_image, step_preprocess
    from src.preprocessing.image_enhancement import remap_to_original
    from src.risk_assessment.risk_engine import RiskEngine

    timer = StageTimer()
    row: dict[str, Any] = {
        "image_path": image_path,
        "category": category,
        "success": False,
        "error": None,
    }

    try:
        with timer.measure("load"):
            load_result = step_load_image(image_path)
            bgr = cv2.imread(load_result["path"])
            if bgr is None:
                raise RuntimeError("OpenCV could not read image for preprocessing.")

        preprocessed_path = None
        try:
            with timer.measure("preprocess"):
                preprocessed_path, prep_result = step_preprocess(bgr)

            with timer.measure("detect"):
                detection_result = step_detect_qr(preprocessed_path)
        finally:
            if preprocessed_path and Path(preprocessed_path).exists():
                Path(preprocessed_path).unlink(missing_ok=True)

        detection_result = remap_to_original(detection_result, prep_result)

        detected = bool(detection_result["detected"])
        qr_data = detection_result["detections"][0]["data"] if detected else None

        with timer.measure("url_analyze"):
            url_result = run_url_analysis(qr_data)

        with timer.measure("risk_assess"):
            engine = RiskEngine()
            risk_result = engine.assess(detection_result, image_id=image_path)

        predicted_malicious = risk_result.risk_level.severity >= risk_severity_threshold

        row.update(
            {
                "success": True,
                "width": load_result["width"],
                "height": load_result["height"],
                "detected": detected,
                "qr_count": detection_result["count"],
                "detector_used": detection_result["detector_used"],
                "qr_data": qr_data,
                "risk_level": risk_result.risk_level.value,
                "risk_score": risk_result.score,
                "confidence": risk_result.confidence,
                "recommendation": risk_result.recommendation,
                "expect_qr": ground_truth["expect_qr"],
                "expect_malicious": ground_truth["expect_malicious"],
                "predicted_malicious": predicted_malicious,
                "timings_ms": timer.as_dict(),
                "total_time_ms": timer.total_ms,
            }
        )
        _merge_url_analysis_fields(row, url_result)
    except Exception as exc:  # noqa: BLE001 — must never crash a worker
        row.update(
            {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "expect_qr": ground_truth["expect_qr"],
                "expect_malicious": ground_truth["expect_malicious"],
                "timings_ms": timer.as_dict(),
                "total_time_ms": timer.total_ms,
            }
        )
        logger.warning("Failed on %s: %s", image_path, row["error"])

    return row


def _merge_url_analysis_fields(row: dict[str, Any], url_result: dict[str, Any]) -> None:
    """Merge URL Analyzer output into *row*, in place — Evaluation Framework Update.

    When ``url_result["available"]`` is falsy (module not integrated yet, or
    no QR payload to analyse), this function does nothing: *row* keeps
    exactly the shape it had before URL Analyzer support existed, which is
    what makes ``results.csv``/``results.json`` fully backward compatible
    when the URL Analyzer isn't present.

    When available, two things are added:

    * ``row["url_analysis"]`` — the full normalised signal dict (for
      ``results.json``; nothing here is dropped).
    * The specific Title-Case columns requested for ``results.csv``
      (``"Decoded URL"``, ``"URL Valid"``, ``"HTTPS"``, ...).
    """
    if not url_result.get("available"):
        return

    row["url_analysis"] = url_result

    if url_result.get("error"):
        return  # ran, but failed — no signal columns to report for this image

    row["Decoded URL"] = url_result.get("decoded_url")
    row["URL Valid"] = url_result.get("url_valid")
    row["HTTPS"] = url_result.get("https")
    row["Shortener"] = url_result.get("is_shortener")
    row["Contains IP"] = url_result.get("contains_ip")
    row["Suspicious Keywords"] = url_result.get("suspicious_keywords")
    row["Suspicious TLD"] = url_result.get("suspicious_tld")
    row["Homograph"] = url_result.get("is_homograph")
    row["Entropy Score"] = url_result.get("entropy_score")
    row["URL Risk Score"] = url_result.get("overall_url_risk")
    row["URL Risk Level"] = url_result.get("url_risk_level")


# ===========================================================================
# Orchestration
# ===========================================================================

def run_evaluation(args: argparse.Namespace) -> int:
    """Coordinate the full evaluation workflow. Returns a process exit code."""
    with resolve_dataset_source(args.dataset_root) as (dataset_root, dataset_name):
        out_root = Path(args.output_dir) / dataset_name
        subdirs = {key: out_root / name for key, name in config.OUTPUT_SUBDIRS.items()}
        for d in subdirs.values():
            d.mkdir(parents=True, exist_ok=True)

        logger_local = setup_logging(
            verbose=args.verbose,
            log_file=args.log_file or (subdirs["logs"] / config.LOG_FILENAME),
        )

        system_info = collect_system_info()
        logger_local.debug("System info: %s", system_info.to_dict())

        labels = load_category_labels(args.labels)

        try:
            records, stats = discover_images(dataset_root, recursive=args.recursive)
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger_local.error(str(exc))
            return 1

        print(stats.pretty_print())

        if not records:
            logger_local.error("No images to evaluate — exiting.")
            return 1

        # ---- Duplicate detection -----------------------------------------------
        if args.duplicate_detection:
            dup_report = find_duplicates(records)
        else:
            from src.evaluation.duplicate_detector import DuplicateReport

            dup_report = DuplicateReport(unique_records=records)
        logger_local.info(
            "Duplicate detection: %d unique image(s), %d duplicate(s) skipped.",
            len(dup_report.unique_records), dup_report.duplicate_count,
        )

        # ---- Checkpoint / resume ------------------------------------------------
        checkpoint_path = subdirs["checkpoints"] / config.CHECKPOINT_FILENAME
        checkpoint = None
        if args.resume:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint is not None:
                logger_local.info(
                    "Resuming from checkpoint: %d image(s) already completed.",
                    len(checkpoint.completed_paths),
                )
        if checkpoint is None:
            checkpoint = Checkpoint(dataset_root=str(dataset_root))

        pending_records = [
            r for r in dup_report.unique_records if not checkpoint.is_done(r.path)
        ]

        worker_count = args.workers or os.cpu_count() or 1
        logger_local.info(
            "Running evaluation with %d worker(s) over %d image(s) (%d already checkpointed).",
            worker_count, len(pending_records), len(checkpoint.completed_paths),
        )

        progress = ProgressReporter(total=len(dup_report.unique_records))
        for _ in checkpoint.completed_paths:
            progress.update("resumed")
        wall_clock_start = time.perf_counter()

        newly_completed = 0
        try:
            if pending_records:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    futures = {
                        executor.submit(
                            process_single_image,
                            record.path,
                            record.category,
                            ground_truth_for(record.category, labels),
                            args.risk_threshold,
                        ): record
                        for record in pending_records
                    }
                    for future in as_completed(futures):
                        record = futures[future]
                        try:
                            row = future.result()
                        except Exception as exc:  # noqa: BLE001 — worker-level crash safety net
                            row = {
                                "image_path": record.path,
                                "category": record.category,
                                "success": False,
                                "error": f"Worker crashed: {exc}",
                                "expect_qr": ground_truth_for(record.category, labels)["expect_qr"],
                                "expect_malicious": ground_truth_for(record.category, labels)["expect_malicious"],
                                "timings_ms": {},
                                "total_time_ms": 0.0,
                            }
                        checkpoint.mark_done(row)
                        newly_completed += 1
                        progress.update(record.category)

                        if should_checkpoint(newly_completed, config.CHECKPOINT_INTERVAL_IMAGES):
                            save_checkpoint(checkpoint, checkpoint_path)

                        if args.fail_fast and not row["success"]:
                            logger_local.error("--fail-fast: aborting after failure on %s", record.path)
                            break
        except KeyboardInterrupt:
            logger_local.warning(
                "Interrupted — saving checkpoint with %d completed image(s). "
                "Re-run with --resume to continue.", len(checkpoint.completed_paths),
            )
        finally:
            progress.finish()
            save_checkpoint(checkpoint, checkpoint_path)

        # ---- Clone results onto duplicates --------------------------------------
        canonical_by_path = {r["image_path"]: r for r in checkpoint.results}
        duplicate_rows: list[dict[str, Any]] = []
        for dup_path, canonical_path in dup_report.duplicate_map.items():
            canonical_row = canonical_by_path.get(canonical_path)
            if canonical_row is None or not canonical_row.get("success"):
                continue
            dup_category = next(
                (rec.category for rec in records if rec.path == dup_path), "uncategorized"
            )
            duplicate_rows.append(
                clone_result_for_duplicates(
                    canonical_row, dup_path, dup_category, ground_truth_for(dup_category, labels)
                )
            )

        results: list[dict[str, Any]] = list(checkpoint.results) + duplicate_rows

        wall_clock_seconds = time.perf_counter() - wall_clock_start
        logger_local.info(
            "Evaluation finished in %s (%d/%d images processed, %d via duplicate reuse).",
            format_duration(wall_clock_seconds), len(checkpoint.results), len(records), len(duplicate_rows),
        )

        return _finalize_and_write_outputs(
            args, results, wall_clock_seconds, worker_count, out_root, subdirs, system_info, dataset_root
        )


def _aggregate_url_analysis(successful: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate per-image URL Analyzer output into metrics/plot-ready data.

    Returns ``None`` when no image in the run produced usable URL Analyzer
    output (module absent, or no QR codes decoded) — the caller uses this
    to skip every URL-related output (CSV columns, JSON section, plots,
    report section) automatically, which is what keeps a run with no URL
    Analyzer byte-for-byte identical to a run that never had URL Analyzer
    support.
    """
    usable = [
        r
        for r in successful
        if r.get("url_analysis", {}).get("available") and not r["url_analysis"].get("error")
    ]
    if not usable:
        return None

    scored = [r for r in usable if r["url_analysis"].get("predicted_malicious") is not None]

    by_category: dict[str, Any] = {}
    grouped: dict[str, list[dict]] = {}
    for r in scored:
        grouped.setdefault(r["category"], []).append(r)
    for category, rows in grouped.items():
        by_category[category] = compute_binary_metrics(
            [r["url_analysis"]["predicted_malicious"] for r in rows],
            [r["expect_malicious"] for r in rows],
        )

    overall = (
        compute_binary_metrics(
            [r["url_analysis"]["predicted_malicious"] for r in scored],
            [r["expect_malicious"] for r in scored],
        )
        if scored
        else None
    )

    risk_scores = [
        float(r["url_analysis"]["overall_url_risk"])
        for r in usable
        if isinstance(r["url_analysis"].get("overall_url_risk"), (int, float))
    ]
    analysis_times = [
        r["timings_ms"]["url_analyze"] for r in usable if "url_analyze" in r.get("timings_ms", {})
    ]
    decoded_count = sum(1 for r in successful if r.get("detected"))

    https_flags = [r["url_analysis"].get("https") for r in usable if r["url_analysis"].get("https") is not None]
    shortener_flags = [
        r["url_analysis"].get("is_shortener") for r in usable if r["url_analysis"].get("is_shortener") is not None
    ]

    keyword_counts: dict[str, int] = {}
    for r in usable:
        for kw in (r["url_analysis"].get("suspicious_keywords") or []):
            keyword_counts[str(kw)] = keyword_counts.get(str(kw), 0) + 1

    tld_counts: dict[str, int] = {}
    for r in usable:
        if r["url_analysis"].get("suspicious_tld") and r["url_analysis"].get("tld"):
            tld_val = str(r["url_analysis"]["tld"])
            tld_counts[tld_val] = tld_counts.get(tld_val, 0) + 1

    return {
        "by_category": by_category,
        "overall": overall,
        "summary_stats": {
            "average_risk_score": compute_numeric_stats(risk_scores).mean,
            "average_analysis_time_ms": compute_numeric_stats(analysis_times).mean,
            "success_rate": (len(usable) / decoded_count) if decoded_count else 0.0,
        },
        "plot_data": {
            "risk_scores": risk_scores,
            "https_count": sum(1 for v in https_flags if v),
            "http_count": sum(1 for v in https_flags if not v),
            "shortener_count": sum(1 for v in shortener_flags if v),
            "non_shortener_count": sum(1 for v in shortener_flags if not v),
            "keyword_counts": keyword_counts,
            "tld_counts": tld_counts,
            "visual_risk_scores": [r["risk_score"] for r in usable],
            "url_risk_scores": risk_scores,
        },
    }


def _finalize_and_write_outputs(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    wall_clock_seconds: float,
    worker_count: int,
    out_root: Path,
    subdirs: dict[str, Path],
    system_info,
    dataset_root: Path,
) -> int:
    """Compute every metric/benchmark once, then hand it to CSV/JSON/plots/reports/gallery."""
    # ---- Detection metrics -------------------------------------------------
    successful = [r for r in results if r["success"]]

    detection_preds = [r["detected"] for r in successful]
    detection_truth = [r["expect_qr"] for r in successful]
    detection_overall = compute_binary_metrics(detection_preds, detection_truth)

    detection_by_category: dict[str, Any] = {}
    risk_by_category: dict[str, Any] = {}
    by_category: dict[str, list[dict]] = {}
    for r in successful:
        by_category.setdefault(r["category"], []).append(r)

    for category, rows in by_category.items():
        detection_by_category[category] = compute_binary_metrics(
            [r["detected"] for r in rows], [r["expect_qr"] for r in rows]
        )
        risk_by_category[category] = compute_binary_metrics(
            [r["predicted_malicious"] for r in rows], [r["expect_malicious"] for r in rows]
        )

    risk_preds = [r["predicted_malicious"] for r in successful]
    risk_truth = [r["expect_malicious"] for r in successful]
    risk_overall = compute_binary_metrics(risk_preds, risk_truth)

    detection_cm = build_confusion_matrix("QR Detection", detection_preds, detection_truth)
    risk_cm = build_confusion_matrix("Risk Classification", risk_preds, risk_truth)

    # ---- Benchmark -----------------------------------------------------------
    benchmark = build_benchmark_report(results, wall_clock_seconds, worker_count)

    # ---- URL Analyzer ----------------------------------------------------------
    # Returns None whenever the URL Analyzer isn't integrated yet (or no QR
    # codes were decoded this run) — every block below checks this before
    # writing anything URL-related.
    url_analysis = _aggregate_url_analysis(successful)

    # ---- CSV outputs ------------------------------------------------------------
    if args.save_csv:
        csv_rows = [{k: v for k, v in r.items() if k != "url_analysis"} for r in results]
        write_csv(csv_rows, subdirs["csv"] / config.RESULTS_CSV_FILENAME)
        write_csv(
            [
                {"category": c, **detection_by_category[c].to_dict()}
                for c in sorted(detection_by_category)
            ],
            subdirs["csv"] / config.CATEGORY_SUMMARY_CSV_FILENAME,
        )
        write_csv(
            [{"category": c, **cb.to_dict()} for c, cb in benchmark.per_category.items()],
            subdirs["csv"] / config.BENCHMARK_CSV_FILENAME,
        )
        if url_analysis and url_analysis["by_category"]:
            write_csv(
                [
                    {"category": c, **m.to_dict()}
                    for c, m in sorted(url_analysis["by_category"].items())
                ],
                subdirs["csv"] / config.URL_SUMMARY_CSV_FILENAME,
            )
        logger.info("CSV outputs written to %s", subdirs["csv"])

    # ---- JSON outputs ---------------------------------------------------------
    if args.save_json:
        write_json(results, subdirs["json"] / config.RESULTS_JSON_FILENAME)
        summary_payload: dict[str, Any] = {
            "detection": {c: m.to_dict() for c, m in detection_by_category.items()},
            "risk": {c: m.to_dict() for c, m in risk_by_category.items()},
            "detection_overall": detection_overall.to_dict(),
            "risk_overall": risk_overall.to_dict(),
        }
        if url_analysis:
            summary_payload["url_analysis"] = {
                "by_category": {c: m.to_dict() for c, m in url_analysis["by_category"].items()},
                **(
                    {"overall": url_analysis["overall"].to_dict()}
                    if url_analysis["overall"] is not None
                    else {}
                ),
                "summary_stats": url_analysis["summary_stats"],
            }
        write_json(summary_payload, subdirs["json"] / config.SUMMARY_JSON_FILENAME)

        benchmark_payload = benchmark.to_dict()
        benchmark_payload["system_info"] = system_info.to_dict()
        write_json(benchmark_payload, subdirs["json"] / config.BENCHMARK_JSON_FILENAME)
        logger.info("JSON outputs written to %s", subdirs["json"])

    # ---- Charts -----------------------------------------------------------------
    plot_paths: list[Path] = []
    url_plot_paths: list[Path] = []
    if args.generate_plots:
        from src.evaluation.plots import generate_all_plots, generate_url_analyzer_plots

        detection_rate_by_category = {c: m.detection_rate for c, m in detection_by_category.items()}
        risk_level_counts: dict[str, int] = {}
        for r in successful:
            risk_level_counts[r["risk_level"]] = risk_level_counts.get(r["risk_level"], 0) + 1
        confidences = [r["confidence"] for r in successful]
        total_times = [r["total_time_ms"] for r in successful]
        fp_by_category = {c: m.false_positives for c, m in detection_by_category.items()}
        fn_by_category = {c: m.false_negatives for c, m in detection_by_category.items()}

        plot_paths = generate_all_plots(
            benchmark=benchmark,
            detection_rate_by_category=detection_rate_by_category,
            detection_cm=detection_cm,
            risk_cm=risk_cm,
            risk_level_counts=risk_level_counts,
            confidences=confidences,
            total_times_ms=total_times,
            fp_by_category=fp_by_category,
            fn_by_category=fn_by_category,
            out_dir=subdirs["charts"],
        )

        if url_analysis:
            pd = url_analysis["plot_data"]
            url_plot_paths = generate_url_analyzer_plots(
                risk_scores=pd["risk_scores"],
                https_count=pd["https_count"],
                http_count=pd["http_count"],
                shortener_count=pd["shortener_count"],
                non_shortener_count=pd["non_shortener_count"],
                keyword_counts=pd["keyword_counts"],
                tld_counts=pd["tld_counts"],
                visual_risk_scores=pd["visual_risk_scores"],
                url_risk_scores_for_scatter=pd["url_risk_scores"],
                out_dir=subdirs["charts"],
            )

    # ---- Reports (markdown + HTML) ---------------------------------------------
    if args.generate_report:
        common_kwargs = dict(
            dataset_root=str(dataset_root),
            detection_metrics_by_category=detection_by_category,
            detection_metrics_overall=detection_overall,
            risk_metrics_by_category=risk_by_category,
            risk_metrics_overall=risk_overall,
            benchmark=benchmark,
            detection_cm=detection_cm,
            risk_cm=risk_cm,
            url_metrics_by_category=(url_analysis["by_category"] if url_analysis else None),
            url_metrics_overall=(url_analysis["overall"] if url_analysis else None),
            url_summary_stats=(url_analysis["summary_stats"] if url_analysis else None),
        )

        markdown = generate_markdown_report(plot_paths=plot_paths, url_plot_paths=url_plot_paths, **common_kwargs)
        report_path = write_report(markdown, subdirs["reports"] / config.MARKDOWN_REPORT_FILENAME)
        logger.info("Markdown report written to %s", report_path)

        html = generate_html_report(
            plot_paths=plot_paths, url_plot_paths=url_plot_paths, system_info=system_info, **common_kwargs
        )
        html_path = write_html_report(html, subdirs["reports"] / config.HTML_REPORT_FILENAME)
        logger.info("HTML report written to %s", html_path)

    # ---- Gallery ----------------------------------------------------------------
    if args.gallery:
        build_galleries(results, subdirs["gallery"])
        copy_failed_images(results, subdirs["failed_images"])

    failures = len(results) - len(successful)
    print(
        f"\nDone. {len(successful)}/{len(results)} images processed successfully "
        f"({failures} failure(s)). Outputs in: {out_root}"
    )
    return 0 if failures == 0 else (0 if not args.fail_fast else 1)


# ===========================================================================
# CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.evaluate_dataset",
        description="Benchmark the QR Shield pipeline over a labelled image dataset.",
    )
    parser.add_argument(
        "dataset_root",
        help="Dataset folder or .zip archive (e.g. data/evaluation or datasets/batch1.zip)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into nested subdirectories (default: on; use --no-recursive to disable)",
    )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write results.json / summary.json / benchmark.json (default: on; use --no-save-json to disable)",
    )
    parser.add_argument(
        "--save-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write results.csv / category_summary.csv / benchmark.csv (default: on; use --no-save-csv to disable)",
    )
    parser.add_argument(
        "--generate-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the chart set under results/<dataset>/charts/ (default: on; use --no-generate-plots to disable)",
    )
    parser.add_argument(
        "--generate-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Evaluation_Report.md and Evaluation_Report.html (default: on; use --no-generate-report to disable)",
    )
    parser.add_argument("--workers", type=int, default=config.DEFAULT_WORKER_COUNT, help="Number of parallel workers (default: auto-detect CPU cores)")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the run on the first per-image failure")
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_BASE_DIR, help="Base directory; outputs are written under <output-dir>/<dataset_name>/")
    parser.add_argument("--labels", default=None, help="Optional JSON file overriding the category ground-truth convention")
    parser.add_argument("--risk-threshold", type=int, default=config.DEFAULT_RISK_SEVERITY_THRESHOLD, help="Minimum RiskLevel severity (0=SAFE,1=SUSPICIOUS,2=HIGH_RISK) counted as predicted-malicious")
    parser.add_argument("--log-file", default=None, help="Optional path overriding the default results/<dataset>/logs/evaluation.log")
    parser.add_argument(
        "--duplicate-detection",
        action=argparse.BooleanOptionalAction,
        default=config.DUPLICATE_DETECTION_ENABLED_DEFAULT,
        help="Skip re-running the pipeline on byte-identical duplicate images (default: on)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from results/<dataset>/checkpoints/resume.json if present",
    )
    parser.add_argument(
        "--gallery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Populate results/<dataset>/gallery/ and failed_images/ (default: on)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.exit(run_evaluation(args))


if __name__ == "__main__":
    main()