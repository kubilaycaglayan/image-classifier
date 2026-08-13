"""Download a small, attributed Wikimedia Commons dataset.

This script intentionally keeps the dataset small for the educational project:
15 training images and 5 validation images per class. Wikimedia Commons media
files carry their own license terms, so the script records the license and
source page for every downloaded image in ``data/provenance.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote, urlencode

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
IMAGES_PER_CLASS = 20
TRAIN_IMAGES_PER_CLASS = 15
THUMBNAIL_WIDTH = 256
USER_AGENT = "image-classifier-educational-project/1.0"

SEARCH_TERMS = {
    "bottle": ("Bottle", "water bottle", "beer bottle", "glass bottle"),
    "pencil": ("pencil", "colored pencil", "pencils"),
    "wristwatch": ("wristwatch", "wrist watch"),
}

ALLOWED_LICENSE_PREFIXES = (
    "CC0",
    "CC BY",
    "CC BY-SA",
    "Public domain",
    "PD",
)


def commons_search(search_term: str) -> list[dict]:
    """Return image records from the Wikimedia Commons search API."""

    query = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{search_term} filetype:bitmap",
            "gsrnamespace": "6",
            "gsrlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": str(THUMBNAIL_WIDTH),
            "format": "json",
            "formatversion": "2",
        }
    )
    api_url = f"https://commons.wikimedia.org/w/api.php?{query}"
    response = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--user-agent",
            USER_AGENT,
            api_url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(response.stdout).get("query", {}).get("pages", [])


def usable_record(page: dict, class_name: str) -> dict | None:
    """Extract a small reusable record when the API response is suitable."""

    title = page.get("title", "")
    info = page.get("imageinfo", [{}])[0]
    metadata = info.get("extmetadata", {})
    license_name = metadata.get("LicenseShortName", {}).get("value", "").strip()
    title_lower = title.lower()
    relevant_terms = {
        "bottle": ("bottle",),
        "pencil": ("pencil",),
        "wristwatch": ("wristwatch", "wrist watch"),
    }[class_name]

    if not info.get("thumburl") or not info.get("mime", "").startswith("image/"):
        return None
    if not any(term in title_lower for term in relevant_terms):
        return None
    if not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
        return None

    page_id = str(page.get("pageid", ""))
    return {
        "class": class_name,
        "title": title.removeprefix("File:"),
        "page_url": f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "download_url": info["thumburl"],
        "license": license_name,
        "artist": metadata.get("Artist", {}).get("value", "").strip(),
        "page_id": page_id,
    }


def collect_records(class_name: str) -> list[dict]:
    records: list[dict] = []
    seen_page_ids: set[str] = set()
    for search_term in SEARCH_TERMS[class_name]:
        for page in commons_search(search_term):
            record = usable_record(page, class_name)
            if record is None or record["page_id"] in seen_page_ids:
                continue
            seen_page_ids.add(record["page_id"])
            records.append(record)
            if len(records) == IMAGES_PER_CLASS:
                return records
    if len(records) < IMAGES_PER_CLASS:
        raise RuntimeError(
            f"Found only {len(records)} usable {class_name} images; "
            f"need {IMAGES_PER_CLASS}."
        )
    return records


def download_image(record: dict, output_path: Path) -> tuple[int, int]:
    """Download, validate, and convert one thumbnail to an RGB JPEG."""

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
                "--user-agent",
                USER_AGENT,
                "--output",
                temporary_file.name,
                record["download_url"],
            ],
            check=True,
        )
        with Image.open(temporary_file.name) as image:
            image.verify()
        with Image.open(temporary_file.name) as image:
            rgb_image = image.convert("RGB")
            size = rgb_image.size
            rgb_image.save(output_path, format="JPEG", quality=85, optimize=True)
    return size


def main() -> None:
    all_provenance: list[dict] = []
    for class_name in SEARCH_TERMS:
        records = collect_records(class_name)
        for index, record in enumerate(records):
            split = "train" if index < TRAIN_IMAGES_PER_CLASS else "validation"
            digest = hashlib.sha1(record["page_id"].encode()).hexdigest()[:12]
            output_path = DATA_ROOT / split / class_name / f"{digest}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            width, height = download_image(record, output_path)
            all_provenance.append(
                {
                    "split": split,
                    "class": class_name,
                    "filename": str(output_path.relative_to(DATA_ROOT)),
                    "width": width,
                    "height": height,
                    "source_page": record["page_url"],
                    "license": record["license"],
                    "artist": record["artist"],
                }
            )
            print(f"Downloaded {output_path.relative_to(PROJECT_ROOT)}")

    provenance_path = DATA_ROOT / "provenance.csv"
    with provenance_path.open("w", newline="") as provenance_file:
        fieldnames = list(all_provenance[0])
        writer = csv.DictWriter(provenance_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_provenance)
    print(f"Wrote provenance for {len(all_provenance)} images to {provenance_path}")


if __name__ == "__main__":
    main()
