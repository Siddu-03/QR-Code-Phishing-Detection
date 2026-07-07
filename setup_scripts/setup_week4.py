from pathlib import Path

# ==========================================================
# QR Shield
# Week 4 Setup Script
# Final Implementation Week
# ==========================================================

# ----------------------------
# New folders
# ----------------------------

folders = [
    # Reporting Module
    "src/reporting",

    # Week 4 Documentation
    "docs/week4",

    # Final Testing Datasets
    "data/test_results",
    "data/benchmark",
    "data/false_positive",
    "data/false_negative",
]

# ----------------------------
# New files
# ----------------------------

files = [

    # ==========================
    # Reporting Module
    # ==========================
    "src/reporting/report_generator.py",
    "src/reporting/report_models.py",
    "src/reporting/json_export.py",
    "src/reporting/audit_log.py",

    # ==========================
    # Documentation
    # ==========================
    "docs/week4/observations.md",
    "docs/week4/test_results.md",
    "docs/week4/benchmark.md",
    "docs/week4/challenges.md",
    "docs/week4/final_notes.md",
]

# ==========================================================
# Create folders
# ==========================================================

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# ==========================================================
# Create files (only if they don't already exist)
# ==========================================================

for file in files:
    path = Path(file)

    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

# ==========================================================
# Output
# ==========================================================

print("=" * 60)
print("✅ QR Shield - Week 4 setup completed successfully!")
print("=" * 60)

print("\n📁 Folders Created:")
for folder in folders:
    print(f"  • {folder}")

print("\n📄 Files Created:")
for file in files:
    print(f"  • {file}")

print("\nWeek 4 Modules")
print("-" * 60)
print("Member 1 : Tamper Analysis Finalization")
print("Member 2 : Risk Assessment Finalization")
print("Member 3 : Reporting & Final Integration")
print("-" * 60)

print("\n⚠ Existing project files were NOT modified.")
print("Ready for Week 4 development.")
print("=" * 60)