"""
QR Shield
Week 4 Setup v2

Creates ONLY the new files and folders introduced in Week 4 v2.

Safe to run multiple times.
Will NEVER overwrite existing files.
"""

from pathlib import Path

# ==========================================================
# NEW FOLDERS
# ==========================================================

FOLDERS = [

    # URL Analyzer Module
    "src/url_analyzer",

    # Evaluation Dataset Structure
    "data/evaluation",
    "data/evaluation/normal",
    "data/evaluation/rotated",
    "data/evaluation/blurred",
    "data/evaluation/low_light",
    "data/evaluation/perspective",
    "data/evaluation/partially_occluded",
    "data/evaluation/damaged",
    "data/evaluation/overlay_attack",
    "data/evaluation/phishing",

    # Evaluation Outputs
    "results",
    "results/json",
    "results/csv",
    "results/plots",

]

# ==========================================================
# NEW FILES
# ==========================================================

FILES = [

    # Package marker
    "src/url_analyzer/__init__.py",

    # URL Analyzer
    "src/url_analyzer/url_analyzer.py",

    "src/url_analyzer/parser.py",

    "src/url_analyzer/validators.py",

    "src/url_analyzer/domain_checks.py",

    "src/url_analyzer/keyword_analysis.py",

    "src/url_analyzer/entropy.py",

    "src/url_analyzer/reputation.py",

    "src/url_analyzer/url_result.py",

    # Keep empty folders in Git
    "results/.gitkeep",
    "results/json/.gitkeep",
    "results/csv/.gitkeep",
    "results/plots/.gitkeep",

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
print("QR Shield - Week 4 Setup v2")
print("=" * 70)

print("\nCreated Folders :", len(created_folders))
for folder in created_folders:
    print("  +", folder)

print("\nExisting Folders :", len(existing_folders))
for folder in existing_folders:
    print("  =", folder)

print("\nCreated Files :", len(created_files))
for file in created_files:
    print("  +", file)

print("\nExisting Files :", len(existing_files))
for file in existing_files:
    print("  =", file)

print("\nWeek 4 Modules")
print("-" * 70)
print("Member 1 : Tamper Analysis")
print("Member 2 : Risk Assessment")
print("Member 3 : URL Analysis + Reporting + Integration")

print("\nGit Branches")
print("-" * 70)
print("feature/week4-tamper-final")
print("feature/week4-risk-final")
print("feature/week4-url-reporting")

print("\nNew Module Added")
print("-" * 70)
print("src/url_analyzer/")
print("├── __init__.py")
print("├── url_analyzer.py")
print("├── parser.py")
print("├── validators.py")
print("├── domain_checks.py")
print("├── keyword_analysis.py")
print("├── entropy.py")
print("├── reputation.py")
print("└── url_result.py")

print("\nNothing was overwritten.")
print("Existing project files remain untouched.")
print("=" * 70)