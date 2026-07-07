"""
QR Shield
Evaluation Framework Setup

Creates the folder and file structure required for the
research-grade Evaluation Framework.

Safe to run multiple times.
Existing files are NEVER overwritten.
"""

from pathlib import Path

# ==========================================================
# FOLDERS
# ==========================================================

FOLDERS = [

    # Evaluation module
    "src/evaluation",

    # Dataset root
    "data/evaluation",

    # Dataset categories
    "data/evaluation/normal",
    "data/evaluation/blurred",
    "data/evaluation/rotated",
    "data/evaluation/low_light",
    "data/evaluation/perspective",
    "data/evaluation/partially_occluded",
    "data/evaluation/damaged",
    "data/evaluation/overlay_attack",
    "data/evaluation/phishing",
    "data/evaluation/url_security",

    # Dataset archives
    "datasets",

    # Results
    "results",
    "results/csv",
    "results/json",
    "results/reports",
    "results/charts",
    "results/gallery",
    "results/gallery/detected",
    "results/gallery/failed",
    "results/gallery/high_risk",
    "results/gallery/tampered",
    "results/failed_images",
    "results/logs",
    "results/checkpoints"

]

# ==========================================================
# FILES
# ==========================================================

FILES = [

    # Package
    "src/evaluation/__init__.py",

    # Core
    "src/evaluation/evaluate_dataset.py",
    "src/evaluation/dataset_loader.py",
    "src/evaluation/utils.py",
    "src/evaluation/benchmark.py",
    "src/evaluation/metrics.py",
    "src/evaluation/plots.py",
    "src/evaluation/generate_report.py",

    # Optional future helpers
    "src/evaluation/checkpoint.py",
    "src/evaluation/progress.py",
    "src/evaluation/system_info.py",
    "src/evaluation/gallery.py",
    "src/evaluation/html_report.py",
    "src/evaluation/duplicate_detector.py",

    # Keep empty folders in Git
    "datasets/.gitkeep",

    "results/.gitkeep",
    "results/csv/.gitkeep",
    "results/json/.gitkeep",
    "results/reports/.gitkeep",
    "results/charts/.gitkeep",
    "results/gallery/.gitkeep",
    "results/gallery/detected/.gitkeep",
    "results/gallery/failed/.gitkeep",
    "results/gallery/high_risk/.gitkeep",
    "results/gallery/tampered/.gitkeep",
    "results/failed_images/.gitkeep",
    "results/logs/.gitkeep",
    "results/checkpoints/.gitkeep",

    # Documentation
    "results/README.md"

]

# ==========================================================
# CREATE FOLDERS
# ==========================================================

created_folders = []
existing_folders = []

for folder in FOLDERS:

    path = Path(folder)

    if path.exists():
        existing_folders.append(folder)
    else:
        path.mkdir(parents=True, exist_ok=True)
        created_folders.append(folder)

# ==========================================================
# CREATE FILES
# ==========================================================

created_files = []
existing_files = []

for file in FILES:

    path = Path(file)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_files.append(file)
        continue

    path.touch()

    created_files.append(file)

# ==========================================================
# SUMMARY
# ==========================================================

print("=" * 70)
print("QR Shield - Evaluation Framework Setup")
print("=" * 70)

print(f"\nFolders Created : {len(created_folders)}")
for folder in created_folders:
    print(f"  + {folder}")

print(f"\nFolders Existing : {len(existing_folders)}")
for folder in existing_folders:
    print(f"  = {folder}")

print(f"\nFiles Created : {len(created_files)}")
for file in created_files:
    print(f"  + {file}")

print(f"\nFiles Existing : {len(existing_files)}")
for file in existing_files:
    print(f"  = {file}")

print("\nEvaluation Framework Structure")
print("-" * 70)

print(r"""
src/
└── evaluation/
    ├── __init__.py
    ├── evaluate_dataset.py
    ├── dataset_loader.py
    ├── utils.py
    ├── benchmark.py
    ├── metrics.py
    ├── plots.py
    ├── generate_report.py
    ├── checkpoint.py
    ├── progress.py
    ├── system_info.py
    ├── gallery.py
    ├── html_report.py
    └── duplicate_detector.py
""")

print("\nResults Generated")
print("-" * 70)

print(r"""
results/

├── csv/
├── json/
├── reports/
├── charts/
├── gallery/
│   ├── detected/
│   ├── failed/
│   ├── high_risk/
│   └── tampered/
├── failed_images/
├── logs/
└── checkpoints/
""")

print("\nDataset Structure")
print("-" * 70)

print(r"""
datasets/
└── *.zip

or

data/
└── evaluation/
    ├── normal/
    ├── blurred/
    ├── rotated/
    ├── low_light/
    ├── perspective/
    ├── partially_occluded/
    ├── damaged/
    ├── overlay_attack/
    ├── phishing/
    └── url_security/
""")

print("\nSetup Complete.")
print("Existing files were NOT modified.")
print("=" * 70)