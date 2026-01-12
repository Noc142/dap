#!/usr/bin/env python3
"""
Convert JPG/JPEG files under an input directory to PNG files in an output directory.
Preserves image dimensions. Recurses subdirectories by default.

Usage:
  python convert_jpgs_to_pngs.py --input ./jpgs --output ./png
"""
import argparse
from pathlib import Path
from PIL import Image


def convert_file(path: Path, input_dir: Path, output_dir: Path):
    rel = path.relative_to(input_dir)
    out_path = output_dir / rel.with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(out_path, format="PNG")


def find_files(input_dir: Path, recursive: bool):
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    files = []
    if recursive:
        for pat in patterns:
            files.extend(input_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(input_dir.glob(pat))
    return sorted(dict.fromkeys(files))


def main():
    parser = argparse.ArgumentParser(description="Convert JPGs to PNGs (preserve size)")
    parser.add_argument("--input", "-i", default="jpgs", help="input directory (default: ./jpgs)")
    parser.add_argument("--output", "-o", default="pngs", help="output directory (default: ./pngs)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="do not recurse into subdirectories")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        raise SystemExit(1)

    files = find_files(input_dir, args.recursive)
    if not files:
        print(f"No JPG/JPEG files found in {input_dir}")
        return

    for p in files:
        try:
            convert_file(p, input_dir, output_dir)
            print(f"Converted: {p} -> {output_dir / p.relative_to(input_dir).with_suffix('.png')}")
        except Exception as e:
            print(f"Failed to convert {p}: {e}")


if __name__ == "__main__":
    main()
