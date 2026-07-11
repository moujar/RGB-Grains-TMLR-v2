#!/usr/bin/env python3
"""Manual by-hand grain tagging (``docs/TODO.md`` #5.5, "Manually").

Two-step, human-in-the-loop workflow for flagging watershed artifacts (dust,
etc.) that the by-size filter in ``rgb_grains/data/cleaning.py`` doesn't
catch:

    # 1. Export every grain crop in a *_processed folder as a JPG:
    python -m rgb_grains.utils.manual_tag export data/perfomix_..._processed reviews/perfomix_review

    # 2. Open reviews/perfomix_review/ in Finder/Preview/whatever and delete
    #    every JPG that looks like dust/debris/an artifact rather than a grain.

    # 3. Apply the review: any .npz whose matching JPG is now missing from
    #    the review folder gets moved into a sibling *_excluded folder
    #    (same convention as rgb_grains.data.cleaning.move_excluded_files):
    python -m rgb_grains.utils.manual_tag apply data/perfomix_..._processed reviews/perfomix_review
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rgb_grains.utils.npz_to_jpg import convert_folder


def export_for_review(processed_dir, review_dir, mode="spectralon"):
    """Convert every ``*.npz`` in *processed_dir* to a JPG in *review_dir*."""
    processed_dir = Path(processed_dir)
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    _, npz_files = convert_folder(processed_dir, review_dir, recursive=False, pattern="*.npz", mode=mode)
    print(f"[manual_tag] Exported {len(npz_files)} grain crop(s) to {review_dir} for review.")
    print("[manual_tag] Delete the JPGs that look like dust/artifacts, then run 'apply'.")
    return npz_files


def apply_review(processed_dir, review_dir, dry_run=False):
    """Move every ``.npz`` in *processed_dir* whose JPG is missing from *review_dir*
    into a sibling ``*_excluded`` folder, and record the exclusion list.
    """
    processed_dir = Path(processed_dir)
    review_dir = Path(review_dir)
    excluded_dir = processed_dir.parent / processed_dir.name.replace("_processed", "_excluded")

    npz_files = sorted(processed_dir.glob("*.npz"))
    if not npz_files:
        print(f"[manual_tag] No .npz files found in {processed_dir}")
        return []

    excluded = []
    for npz_path in npz_files:
        jpg_path = review_dir / npz_path.with_suffix(".jpg").name
        if jpg_path.exists():
            continue
        excluded.append(npz_path)
        if dry_run:
            print(f"  [manual_tag] would exclude {npz_path.name} (JPG deleted from review) -> {excluded_dir.name}/")
        else:
            excluded_dir.mkdir(exist_ok=True)
            shutil.move(str(npz_path), str(excluded_dir / npz_path.name))

    if not dry_run and excluded:
        list_path = review_dir / "manual_exclusion_list.txt"
        list_path.write_text("\n".join(str(p) for p in excluded) + "\n")
        print(f"[manual_tag] Exclusion list written to {list_path}")

    print(
        f"[manual_tag] {len(excluded)}/{len(npz_files)} grain(s) {'would be' if dry_run else 'were'} excluded "
        f"(JPG missing from {review_dir})."
    )
    return excluded


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Convert a *_processed folder's grain crops to JPGs for manual review.")
    exp.add_argument("processed_dir", type=str, help="Path to a *_processed folder of grain .npz files.")
    exp.add_argument("review_dir", type=str, help="Output folder for review JPGs (delete bad ones from here).")
    exp.add_argument("--mode", choices=["minmax", "spectralon"], default="spectralon",
                      help="JPG rendering mode, see rgb_grains.utils.npz_to_jpg (default: spectralon).")

    app = sub.add_parser("apply", help="Move .npz files whose JPG was deleted from review_dir into *_excluded.")
    app.add_argument("processed_dir", type=str)
    app.add_argument("review_dir", type=str)
    app.add_argument("--dry-run", action="store_true", help="Report what would be excluded without moving anything.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "export":
        export_for_review(args.processed_dir, args.review_dir, mode=args.mode)
    else:
        apply_review(args.processed_dir, args.review_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
