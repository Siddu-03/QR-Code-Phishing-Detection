from pathlib import Path

# New folders
folders = [
    "src/preprocessing",
    "src/tamper_analysis",
    "docs/week2",
    "docs/research_material",
    "data/normal",
    "data/rotated",
    "data/low_light",
    "data/blurry",
    "data/overlay",
    "data/barcode",
]

# New files
files = [
    "src/preprocessing/image_enhancement.py",
    "src/qr_detector/qr_enhancement.py",
    "src/tamper_analysis/edge_detection.py",
    "src/tamper_analysis/contour_analysis.py",
    "src/tamper_analysis/overlay_detection.py",
    "src/image_loader/dataset_manager.py",
    "docs/week2/observations.md",
    "docs/week2/test_results.md",
    "docs/week2/challenges.md",
]

# Create folders
for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Create files if they don't exist
for file in files:
    path = Path(file)
    if not path.exists():
        path.touch()

print("✅ Week 2 folder structure created successfully!")

