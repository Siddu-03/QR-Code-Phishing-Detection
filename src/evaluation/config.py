"""
config.py
=========
Evaluation Framework — centralized configuration.

Every constant the rest of ``src/evaluation`` needs is defined here:
supported file extensions, default paths, default worker/refresh/interval
counts, chart export formats, and the category ground-truth convention.
No business logic lives in this file — only names and defaults that other
modules import, per the "centralize all configuration values inside
config.py" rule. ``utils.py`` holds the *functions* that operate on this
data (e.g. loading a ``labels.json`` override); this file holds only the
data itself.
"""

from __future__ import annotations

# ===========================================================================
# Supported inputs
# ===========================================================================

#: Image file extensions the dataset loader will treat as evaluable images.
SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)

#: Archive extensions the dataset loader will transparently extract before
#: discovery. Extraction happens into a managed temporary directory that is
#: cleaned up after the run (see ``dataset_loader.extract_if_archive``).
SUPPORTED_ARCHIVE_EXTENSIONS: frozenset[str] = frozenset({".zip"})

# ===========================================================================
# Default paths & runtime knobs
# ===========================================================================

#: Default dataset root used by the CLI when none is given explicitly.
DEFAULT_DATASET_DIR: str = "data/evaluation"

#: Base directory under which every run's ``<dataset_name>/`` folder is created.
DEFAULT_OUTPUT_BASE_DIR: str = "results"

#: 0 tells the orchestrator to auto-detect via ``os.cpu_count()``.
DEFAULT_WORKER_COUNT: int = 0

#: Risk-level severity (0=SAFE, 1=SUSPICIOUS, 2=HIGH_RISK) at/above which a
#: RiskEngine result counts as "predicted malicious" for the risk-classification
#: confusion matrix.
DEFAULT_RISK_SEVERITY_THRESHOLD: int = 1

#: Normalised (0-1) URL-risk score at/above which a URL Analyzer result counts
#: as "predicted malicious" when the analyzer supplies a numeric score but no
#: categorical risk level. See ``url_analyzer_adapter.py``.
DEFAULT_URL_RISK_SCORE_THRESHOLD: float = 0.5

# ===========================================================================
# Progress reporting
# ===========================================================================

#: Minimum wall-clock seconds between two progress-bar redraws, to avoid
#: flooding the terminal on very fast (e.g. tiny-dataset) runs.
PROGRESS_REFRESH_INTERVAL_SECONDS: float = 0.1

PROGRESS_BAR_WIDTH: int = 30

# ===========================================================================
# Checkpointing
# ===========================================================================

#: Write (or overwrite) the checkpoint file after this many newly completed
#: images, in addition to always writing one at the end of a run.
CHECKPOINT_INTERVAL_IMAGES: int = 25

CHECKPOINT_FILENAME: str = "resume.json"

# ===========================================================================
# Duplicate detection
# ===========================================================================

#: Hash algorithm used to fingerprint image *file bytes* for exact-duplicate
#: detection (cheap, deterministic, no extra dependency).
DUPLICATE_HASH_ALGORITHM: str = "sha256"

#: When True, an image whose content hash matches one already seen in this
#: run is not re-run through the pipeline; its result is copied from the
#: first occurrence instead.
DUPLICATE_DETECTION_ENABLED_DEFAULT: bool = True

# ===========================================================================
# Gallery
# ===========================================================================

#: Gallery subfolder names created under ``results/<dataset>/gallery/``.
GALLERY_CATEGORIES: tuple[str, ...] = ("detected", "failed", "high_risk", "tampered")

#: Maximum images copied into each gallery subfolder (keeps the results
#: directory bounded for very large datasets); ``None`` disables the cap.
GALLERY_MAX_IMAGES_PER_CATEGORY: int | None = 200

#: Risk levels (from ``risk_result.RiskLevel``) that route an image into the
#: "high_risk" gallery.
GALLERY_HIGH_RISK_LEVELS: frozenset[str] = frozenset({"HIGH_RISK"})

#: Risk levels that route an image into the "tampered" gallery. Distinct from
#: high-risk today (same underlying levels) but kept as its own constant since
#: Tamper Analysis output may refine this independently in the future.
GALLERY_TAMPERED_LEVELS: frozenset[str] = frozenset({"HIGH_RISK", "SUSPICIOUS"})

# ===========================================================================
# Charts
# ===========================================================================

#: Export formats produced for every generated chart. ``"png"`` is always
#: produced (used for markdown/HTML embedding); add ``"svg"``/``"pdf"`` here
#: to also export those formats.
CHART_EXPORT_FORMATS: tuple[str, ...] = ("png",)

CHART_DPI: int = 150

# ===========================================================================
# Output directory layout
# ===========================================================================

#: Subdirectory names created under ``results/<dataset_name>/``.
OUTPUT_SUBDIRS: dict[str, str] = {
    "csv": "csv",
    "json": "json",
    "reports": "reports",
    "charts": "charts",
    "gallery": "gallery",
    "failed_images": "failed_images",
    "logs": "logs",
    "checkpoints": "checkpoints",
}

RESULTS_CSV_FILENAME: str = "results.csv"
BENCHMARK_CSV_FILENAME: str = "benchmark.csv"
CATEGORY_SUMMARY_CSV_FILENAME: str = "category_summary.csv"
URL_SUMMARY_CSV_FILENAME: str = "url_analysis_summary.csv"

RESULTS_JSON_FILENAME: str = "results.json"
SUMMARY_JSON_FILENAME: str = "summary.json"
BENCHMARK_JSON_FILENAME: str = "benchmark.json"

MARKDOWN_REPORT_FILENAME: str = "Evaluation_Report.md"
HTML_REPORT_FILENAME: str = "Evaluation_Report.html"

LOG_FILENAME: str = "evaluation.log"

# ===========================================================================
# Ground-truth convention (category folder name -> expected labels)
# ===========================================================================

#: See ``utils.load_category_labels`` for how this is loaded/overridden.
DEFAULT_CATEGORY_LABELS: dict[str, dict[str, bool]] = {
    "normal":              {"expect_qr": True, "expect_malicious": False},
    "rotated":             {"expect_qr": True, "expect_malicious": False},
    "blurred":             {"expect_qr": True, "expect_malicious": False},
    "low_light":           {"expect_qr": True, "expect_malicious": False},
    "perspective":         {"expect_qr": True, "expect_malicious": False},
    "partially_occluded":  {"expect_qr": True, "expect_malicious": False},
    "damaged":             {"expect_qr": True, "expect_malicious": False},
    "overlay_attack":      {"expect_qr": True, "expect_malicious": True},
    "phishing":            {"expect_qr": True, "expect_malicious": True},
}

#: Fallback used for any category not present in the mapping above.
FALLBACK_CATEGORY_LABEL: dict[str, bool] = {"expect_qr": True, "expect_malicious": False}

# ===========================================================================
# Benchmark stage names
# ===========================================================================

#: Pipeline stage names timed for every image, in pipeline order. Extending
#: this tuple (e.g. once Tamper Analysis is wired in) is the only change
#: needed for ``benchmark.py`` to start reporting a new stage's statistics.
PIPELINE_STAGE_NAMES: tuple[str, ...] = (
    "load",
    "preprocess",
    "detect",
    "url_analyze",
    "risk_assess",
    "report",
)

# ===========================================================================
# Experiment management
# ===========================================================================
#
# ``OUTPUT_SUBDIRS`` above still describes every subfolder name produced for
# a dataset. Experiment management (``evaluate_dataset.py`` only) splits
# those keys into two groups without changing any of the folder *names*:
#
# * ``PERSISTENT_SUBDIR_KEYS`` — created once directly under
#   ``results/<dataset_name>/`` and reused/appended-to across every run
#   (``logs/evaluation.log`` keeps appending; ``checkpoints/resume.json``
#   keeps being overwritten — neither behavior changes).
# * ``RUN_SUBDIR_KEYS`` — research artifacts that must never be overwritten
#   between runs, so each run gets its own copy of these subfolders inside
#   a timestamped experiment directory.

#: Subdirectory keys (from ``OUTPUT_SUBDIRS``) that live once per dataset,
#: directly under ``results/<dataset_name>/``, and are shared/cumulative
#: across every experiment run.
PERSISTENT_SUBDIR_KEYS: tuple[str, ...] = ("logs", "checkpoints")

#: Subdirectory keys (from ``OUTPUT_SUBDIRS``) that must be produced fresh,
#: without overwriting prior runs, inside each timestamped experiment
#: directory under ``results/<dataset_name>/runs/<run_id>/``.
RUN_SUBDIR_KEYS: tuple[str, ...] = (
    "csv", "json", "reports", "charts", "gallery", "failed_images",
)

#: Name of the folder (under ``results/<dataset_name>/``) that holds every
#: timestamped experiment run.
RUNS_DIRNAME: str = "runs"

#: Name of the folder (under ``results/<dataset_name>/``) that always
#: mirrors the most recently completed experiment run, for tooling/scripts
#: that want "the latest results" without knowing the run's timestamp.
LATEST_DIRNAME: str = "latest"

#: Timestamp format embedded in every run directory name:
#: ``<dataset_name>_<YYYYMMDD_HHMMSS>``.
RUN_TIMESTAMP_FORMAT: str = "%Y%m%d_%H%M%S"

#: Filename of the per-run metadata document written into every experiment
#: directory (and mirrored into ``latest/``).
EXPERIMENT_METADATA_FILENAME: str = "experiment.json"

#: Version identifiers stamped into every ``experiment.json``. These are
#: independent of the Python package/module versions and simply track the
#: evaluation-framework and QR Shield pipeline revisions used to produce a
#: given experiment, for reproducibility when comparing historical runs.
EVALUATION_FRAMEWORK_VERSION: str = "1.1.0"
PIPELINE_VERSION: str = "1.0.0"