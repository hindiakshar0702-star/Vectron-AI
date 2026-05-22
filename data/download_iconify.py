"""
Download SVG icons from Iconify (https://iconify.design).

Iconify aggregates 200+ open-source icon sets (Material, Lucide, Tabler, Heroicons,
Phosphor, Carbon, etc.) under permissive licenses (mostly MIT / Apache / OFL).

Endpoints used:
- GET https://api.iconify.design/collections
    -> dict of { prefix: {name, license, total, ...} }
- GET https://api.iconify.design/{prefix}.json
    -> bulk JSON containing all icons in the collection.

We persist:
- One .svg file per icon under data/raw/{prefix}/{name}.svg
- A metadata.jsonl with { prefix, name, license, category, tags, viewBox }

Usage:
    python data/download_iconify.py --output data/raw --max-icons 50000
    python data/download_iconify.py --collections mdi,lucide,heroicons --output data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import jsonlines
import requests
from tqdm import tqdm


API_BASE = "https://api.iconify.design"

# Iconify publishes complete collection JSON files via jsDelivr / GitHub.
# The api.iconify.design "/{prefix}.json" endpoint requires an `?icons=...`
# parameter and does NOT return the full collection without it; using these
# CDN URLs is the official way to get all icons in one request.
COLLECTION_JSON_URLS = [
    "https://cdn.jsdelivr.net/npm/@iconify/json@latest/json/{prefix}.json",
    "https://cdn.jsdelivr.net/gh/iconify/icon-sets@master/json/{prefix}.json",
    "https://raw.githubusercontent.com/iconify/icon-sets/master/json/{prefix}.json",
]

DEFAULT_COLLECTIONS = [
    "mdi",          # Material Design Icons (~7K)
    "lucide",       # Lucide (~1.5K)
    "tabler",       # Tabler Icons (~5K)
    "heroicons",    # Heroicons (~300)
    "ph",           # Phosphor (~9K)
    "carbon",       # IBM Carbon (~2K)
    "fluent",       # Microsoft Fluent (~14K)
    "solar",        # Solar (~7K)
    "ic",           # Google Material (~10K)
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_iconify")


def fetch_collections() -> dict:
    """Returns metadata for every collection (name, total, license)."""
    r = requests.get(f"{API_BASE}/collections", timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected /collections response: {type(data).__name__}")
    return data


def fetch_collection(prefix: str) -> dict:
    """Fetch the full Iconify collection JSON for `prefix`.

    Tries several mirrors (jsDelivr -> GitHub raw) because the api.iconify.design
    endpoint requires an `?icons=...` filter and won't return the whole set.
    Always returns a dict on success; raises RuntimeError if every mirror fails
    or the response is not a JSON object containing an `icons` field.
    """
    last_error: Exception | None = None
    for url_template in COLLECTION_JSON_URLS:
        url = url_template.format(prefix=prefix)
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.debug("  mirror failed: %s -> %s", url, e)
            continue

        if not isinstance(data, dict):
            last_error = RuntimeError(
                f"unexpected response type {type(data).__name__} from {url}"
            )
            continue
        if "icons" not in data or not isinstance(data["icons"], dict):
            last_error = RuntimeError(f"response from {url} is missing 'icons' dict")
            continue

        return data

    raise RuntimeError(
        f"could not fetch collection '{prefix}' from any mirror; last error: {last_error}"
    )


def build_svg(icon_body: str, width: int, height: int, view_box: str | None = None) -> str:
    """Wrap an Iconify icon `body` (raw SVG inner) into a complete SVG document."""
    vb = view_box or f"0 0 {width} {height}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{vb}" width="{width}" height="{height}">'
        f"{icon_body}</svg>"
    )


def iter_icons(coll: dict) -> Iterable[tuple[str, str, dict]]:
    """Yield (name, svg_string, meta) for each icon in a collection JSON."""
    if not isinstance(coll, dict):
        return

    icons = coll.get("icons", {}) or {}
    aliases = coll.get("aliases", {}) or {}
    width = coll.get("width", 24)
    height = coll.get("height", 24)
    categories = coll.get("categories", {}) or {}

    if not isinstance(icons, dict):
        return

    name_to_category: dict[str, str] = {}
    if isinstance(categories, dict):
        for cat, names in categories.items():
            if isinstance(names, list):
                for n in names:
                    name_to_category[n] = cat

    for name, data in icons.items():
        if not isinstance(data, dict):
            continue
        body = data.get("body")
        if not body:
            continue
        w = data.get("width", width)
        h = data.get("height", height)
        view_box = f"0 0 {w} {h}"
        svg = build_svg(body, w, h, view_box)
        meta = {
            "name": name,
            "category": name_to_category.get(name, ""),
            "width": w,
            "height": h,
        }
        yield name, svg, meta

    if isinstance(aliases, dict):
        for alias, data in aliases.items():
            if not isinstance(data, dict):
                continue
            parent = data.get("parent")
            if parent in icons and isinstance(icons[parent], dict):
                body = icons[parent].get("body")
                if body:
                    w = icons[parent].get("width", width)
                    h = icons[parent].get("height", height)
                    yield alias, build_svg(body, w, h), {
                        "name": alias,
                        "alias_of": parent,
                        "width": w,
                        "height": h,
                    }


def download(
    collections: list[str],
    output_dir: Path,
    max_icons: int | None,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.jsonl"

    all_collections = fetch_collections()
    total_written = 0

    with jsonlines.open(meta_path, mode="a") as writer:
        for prefix in collections:
            if prefix not in all_collections:
                log.warning("Collection '%s' not found on Iconify, skipping.", prefix)
                continue

            info = all_collections[prefix]
            license_obj = info.get("license", {})
            license_name = license_obj.get("title", "unknown")
            log.info(
                "Fetching '%s' (%s) - %d icons - license: %s",
                prefix,
                info.get("name", prefix),
                info.get("total", 0),
                license_name,
            )

            try:
                coll = fetch_collection(prefix)
            except Exception as e:  # noqa: BLE001
                log.error("Failed to fetch %s: %s -> skipping", prefix, e)
                continue

            coll_dir = output_dir / prefix
            coll_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            icons_dict = coll.get("icons", {}) if isinstance(coll, dict) else {}
            for name, svg, meta in tqdm(
                iter_icons(coll),
                desc=prefix,
                total=len(icons_dict),
                leave=False,
            ):
                svg_path = coll_dir / f"{name}.svg"
                if skip_existing and svg_path.exists():
                    continue
                svg_path.write_text(svg, encoding="utf-8")

                writer.write({
                    "prefix": prefix,
                    "collection_name": info.get("name", prefix),
                    "license": license_name,
                    "path": str(svg_path.relative_to(output_dir)),
                    **meta,
                })
                count += 1
                total_written += 1
                if max_icons and total_written >= max_icons:
                    log.info("Reached max-icons=%d, stopping.", max_icons)
                    return

            log.info("  -> wrote %d icons from %s", count, prefix)

    log.info("Done. Total icons written: %d", total_written)
    log.info("Metadata: %s", meta_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")
    p.add_argument(
        "--collections",
        type=str,
        default=",".join(DEFAULT_COLLECTIONS),
        help="Comma-separated Iconify prefixes",
    )
    p.add_argument("--max-icons", type=int, default=None, help="Stop after N icons total")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--list-collections", action="store_true", help="List all available collections and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_collections:
        cols = fetch_collections()
        rows = [
            (k, v.get("name", k), v.get("total", 0), v.get("license", {}).get("title", "?"))
            for k, v in cols.items()
        ]
        rows.sort(key=lambda r: -r[2])
        for prefix, name, total, lic in rows[:80]:
            print(f"  {prefix:20s}  {total:6d}  {lic:30s}  {name}")
        return

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    download(
        collections=collections,
        output_dir=args.output,
        max_icons=args.max_icons,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
