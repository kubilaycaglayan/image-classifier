"""Populate the educational dataset from public GitHub image collections.

The source repositories and paths are recorded in ``data/provenance.csv``.
This script downloads only 20 images per class (15 train, 5 validation),
converts them to small RGB JPEGs, and refuses to exceed the project limit.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGES_PER_CLASS = 20
TRAIN_IMAGES_PER_CLASS = 15
THUMBNAIL_SIZE = 512

SOURCES = {
    "bottle": [
        ("llsrwsaint/bottlecap-train-dataset-Image-Dataset", "main", "images/")
    ],
    "pencil": [
        ("JSini/pen_pencil_image_dataset", "main", "data/train/pencil/"),
        ("JSini/pen_pencil_image_dataset", "main", "data/test/pencil/"),
        ("yurui777/Data-Pencil", "master", "image_1/"),
    ],
    "wristwatch": [
        ("vedantg10/Chrono_analyzer", "main", "code/images/")
    ],
}


def repository_license(repository: str) -> str:
    """Return the repository's declared SPDX license or an explicit warning."""

    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/license", "--jq", ".license.spdx_id"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "UNSPECIFIED_UPSTREAM_LICENSE"
    license_name = result.stdout.strip()
    return license_name or "UNSPECIFIED_UPSTREAM_LICENSE"


def image_paths(repository: str, branch: str, prefix: str) -> list[str]:
    """List image files beneath a GitHub repository directory."""

    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/git/trees/{branch}?recursive=1",
            "--jq",
            ".tree[] | select(.type == \"blob\") | .path",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    extensions = (".jpg", ".jpeg", ".png", ".webp")
    return [
        path
        for path in result.stdout.splitlines()
        if path.startswith(prefix) and path.lower().endswith(extensions)
    ][:IMAGES_PER_CLASS]


def download_and_convert(url: str, output_path: Path) -> tuple[int, int]:
    """Download one image through curl and save a compact RGB JPEG."""

    with tempfile.NamedTemporaryFile(suffix=".download") as temporary_file:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--output",
                temporary_file.name,
                url,
            ],
            check=True,
        )
        with Image.open(temporary_file.name) as image:
            image = image.convert("RGB")
            image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
            dimensions = image.size
            image.save(output_path, format="JPEG", quality=85, optimize=True)
    return dimensions


def main() -> None:
    provenance: list[dict[str, str | int]] = []
    for class_name, source_list in SOURCES.items():
        source_paths: list[tuple[str, str, str]] = []
        for repository, branch, prefix in source_list:
            source_paths.extend(
                (repository, branch, source_path)
                for source_path in image_paths(repository, branch, prefix)
            )
            if len(source_paths) >= IMAGES_PER_CLASS:
                break
        if len(source_paths) < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"Configured sources supplied {len(source_paths)} images for {class_name}; "
                f"need {IMAGES_PER_CLASS}."
            )
        for index, (repository, branch, source_path) in enumerate(
            source_paths[:IMAGES_PER_CLASS]
        ):
            split = "train" if index < TRAIN_IMAGES_PER_CLASS else "validation"
            filename = f"{index + 1:02d}.jpg"
            output_path = DATA_ROOT / split / class_name / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            width, height = download_and_convert(
                f"https://raw.githubusercontent.com/{repository}/{branch}/{source_path}",
                output_path,
            )
            provenance.append(
                {
                    "split": split,
                    "class": class_name,
                    "filename": str(output_path.relative_to(DATA_ROOT)),
                    "width": width,
                    "height": height,
                    "source_repository": f"https://github.com/{repository}",
                    "source_path": source_path,
                    "declared_repository_license": repository_license(repository),
                }
            )
            print(f"Downloaded {output_path.relative_to(PROJECT_ROOT)}")

    provenance_path = DATA_ROOT / "provenance.csv"
    with provenance_path.open("w", newline="") as provenance_file:
        fieldnames = list(provenance[0])
        writer = csv.DictWriter(provenance_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance)
    print(f"Wrote {len(provenance)} provenance records to {provenance_path}")


if __name__ == "__main__":
    main()
