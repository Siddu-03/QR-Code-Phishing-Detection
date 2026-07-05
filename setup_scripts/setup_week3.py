from pathlib import Path

# ==========================================================
# Week 3 Folder Structure
# Project: QR Shield
# ==========================================================

folders = [
    # Risk Assessment
    "src/risk_assessment",

    # Reporting
    "src/reporting",

    # Week 3 Documentation
    "docs/week3",

    # Testing Dataset
    "data/tampered",
    "data/overlay",
    "data/damaged",
    "data/partial",
    "data/sticker",
]

files = [

    # -----------------------------
    # Member 1 - Tamper Detection
    # -----------------------------
    "src/tamper_analysis/tamper_detector.py",
    "src/tamper_analysis/tamper_result.py",

    # -----------------------------
    # Member 2 - Risk Assessment
    # -----------------------------
    "src/risk_assessment/risk_engine.py",
    "src/risk_assessment/rule_engine.py",
    "src/risk_assessment/scoring.py",
    "src/risk_assessment/risk_result.py",

    # -----------------------------
    # Member 3 - Reporting
    # -----------------------------
    "src/reporting/report_generator.py",
    "src/reporting/json_export.py",
    "src/reporting/audit_log.py",
    "src/reporting/report_models.py",

    # -----------------------------
    # Documentation
    # -----------------------------
    "docs/week3/observations.md",
    "docs/week3/test_results.md",
    "docs/week3/challenges.md",
    "docs/week3/progress.md",
]

# ==========================================================
# Create folders
# ==========================================================

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# ==========================================================
# Create files
# ==========================================================

for file in files:
    path = Path(file)

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

print("=" * 55)
print("✅ Week 3 folder structure created successfully!")
print("=" * 55)
print("\nNew folders created:")
for folder in folders:
    print(f"  📁 {folder}")

print("\nNew files created:")
for file in files:
    print(f"  📄 {file}")

print("\nWeek 3 Modules")
print("----------------------------")
print("Member 1 : Tamper Detection")
print("Member 2 : Risk Assessment")
print("Member 3 : Reporting & Integration")
print("=" * 55)