from pathlib import Path

ROOT = Path("tools") / "dataset_generator"

FILES = [
    "__init__.py",
    "config.py",
    "generator.py",
    "qr_generator.py",
    "url_generator.py",
    "transforms.py",
    "metadata.py",
    "dataset_builder.py",
    "validator.py",
    "utils.py",
    "cli.py",
    "README.md",
]

DIRECTORIES = [
    "templates",
    "output",
    "logs",
    "examples",
]

TEMPLATE_FILES = [
    "templates/benign_urls.txt",
    "templates/malicious_urls.txt",
    "templates/config_template.json",
]

EXAMPLE_FILES = [
    "examples/basic_generation.py",
    "examples/advanced_generation.py",
]


def create_file(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        print(f"[CREATE] {path}")
    else:
        print(f"[SKIP]   {path}")


def main():

    ROOT.mkdir(parents=True, exist_ok=True)

    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    for file in FILES:
        create_file(ROOT / file)

    for file in TEMPLATE_FILES:
        create_file(ROOT / file)

    for file in EXAMPLE_FILES:
        create_file(ROOT / file)

    print("\nDataset Generator structure created successfully.")


if __name__ == "__main__":
    main()