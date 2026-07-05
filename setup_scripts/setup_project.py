from pathlib import Path

# Create folders
folders = [
    "src/image_loader",
    "src/qr_detector",
    "src/visualization",
    "src/integration",

    "data/normal",
    "data/blurry",
    "data/low_light",
    "data/rotated",
    "data/overlay",

    "output/detected",
    "output/cropped",

    "tests",

    "docs/literature-survey",
    "docs/architecture"
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

# Create files with starter content
files = {
    "README.md": "# QR Tamper Detection\n\nComputer Vision based QR Code Tamper Detection System.\n",

    "requirements.txt": """opencv-python
pyzbar
pillow
numpy
streamlit
""",

    ".gitignore": """__pycache__/
*.pyc
.venv/
venv/
.env
""",

    "src/image_loader/image_loader.py": "",

    "src/qr_detector/qr_detector.py": "",

    "src/visualization/draw_box.py": "",

    "src/integration/main.py": """def main():
    print("QR Tamper Detection System")

if __name__ == "__main__":
    main()
"""
}

for filepath, content in files.items():
    file = Path(filepath)
    file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "w", encoding="utf-8") as f:
        f.write(content)

print("Project structure created successfully!")