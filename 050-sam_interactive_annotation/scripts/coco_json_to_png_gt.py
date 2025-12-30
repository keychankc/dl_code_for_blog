#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert COCO instances_*.json to per-image PNG GT masks for this project.

Output format:
- For each image file in img_dir, write a GT mask PNG to out_dir:
  out_dir/{image_stem}.png
- Mask is 0/255 (uint8), foreground=255, background=0.

Modes:
- instance (default): pick ONE instance per image (largest area by default)
- union: union all instances (optionally filtered by category) into one mask

Install:
  pip install pycocotools opencv-python

Example:
  python scripts/coco_json_to_png_gt.py \
    --img_dir data/images \
    --ann_json /path/to/instances_val2017.json \
    --out_dir data/gts \
    --mode instance \
    --pick largest

Union by category name (e.g. person):
  python scripts/coco_json_to_png_gt.py \
    --img_dir data/images \
    --ann_json /path/to/instances_val2017.json \
    --out_dir data/gts \
    --mode union \
    --categories person

Notes:
- This script matches by file_name (e.g., 000000000776.jpg) and writes 000000000776.png
- If an image is not found in JSON, it will be skipped.
"""

import os
import argparse
from typing import List, Optional, Dict

import numpy as np
import cv2
from pycocotools.coco import COCO


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def load_image_list(img_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    return sorted(files)


def categories_to_ids(coco: COCO, cat_names: List[str]) -> List[int]:
    ids = coco.getCatIds(catNms=cat_names)
    return ids


def ann_to_mask(coco: COCO, ann: Dict, h: int, w: int) -> np.ndarray:
    """
    Convert one COCO annotation to a binary mask (0/1).
    """
    m = coco.annToMask(ann)  # (H,W) 0/1
    if m.shape[0] != h or m.shape[1] != w:
        # Very rare, but keep safe:
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, help="Directory with images (subset of COCO val2017, etc.)")
    ap.add_argument("--ann_json", required=True, help="COCO instances JSON, e.g. instances_val2017.json")
    ap.add_argument("--out_dir", required=True, help="Output directory for GT PNG masks (0/255)")
    ap.add_argument("--mode", choices=["instance", "union"], default="instance",
                    help="instance: one instance per image; union: union all instances (optionally by category)")
    ap.add_argument("--pick", choices=["largest", "first"], default="largest",
                    help="When mode=instance: how to choose a single instance")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Optional category names, e.g. person dog. Only used for mode=union "
                         "or to filter instance candidates if provided.")
    ap.add_argument("--min_area", type=float, default=0.0,
                    help="Filter out instances with area < min_area (in pixels, COCO 'area' field).")
    ap.add_argument("--max_images", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    coco = COCO(args.ann_json)

    img_files = load_image_list(args.img_dir)
    if args.max_images and args.max_images > 0:
        img_files = img_files[: args.max_images]

    # Build mapping from COCO file_name -> image_id
    name_to_id = {}
    for img in coco.dataset.get("images", []):
        name_to_id[img["file_name"]] = img["id"]

    # Optional category filter
    cat_ids: Optional[List[int]] = None
    if args.categories:
        cat_ids = categories_to_ids(coco, args.categories)
        if not cat_ids:
            raise ValueError(f"No category ids found for names: {args.categories}")

    written = 0
    skipped_not_found = 0
    skipped_no_ann = 0

    for f in img_files:
        if f not in name_to_id:
            # Image might be from a different split or renamed
            skipped_not_found += 1
            continue

        img_id = name_to_id[f]
        img_info = coco.loadImgs([img_id])[0]
        h, w = int(img_info["height"]), int(img_info["width"])

        # Collect annotations
        if cat_ids is None:
            ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        else:
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

        anns = coco.loadAnns(ann_ids)

        # area filter
        if args.min_area > 0:
            anns = [a for a in anns if float(a.get("area", 0.0)) >= args.min_area]

        if len(anns) == 0:
            skipped_no_ann += 1
            continue

        if args.mode == "instance":
            # If categories provided, it already filtered candidates above
            if args.pick == "largest":
                anns = sorted(anns, key=lambda a: float(a.get("area", 0.0)), reverse=True)
                chosen = anns[0]
            else:
                chosen = anns[0]

            m01 = ann_to_mask(coco, chosen, h, w)

        else:  # union
            m01 = np.zeros((h, w), dtype=np.uint8)
            for a in anns:
                m01 |= ann_to_mask(coco, a, h, w)

        out_path = os.path.join(args.out_dir, f"{stem(f)}.png")
        cv2.imwrite(out_path, (m01 * 255).astype(np.uint8))
        written += 1

    print("Done.")
    print(f"Written masks: {written} -> {args.out_dir}")
    print(f"Skipped (image not in JSON): {skipped_not_found}")
    print(f"Skipped (no annotations after filters): {skipped_no_ann}")
    if args.categories:
        print(f"Category filter: {args.categories}")


if __name__ == "__main__":
    main()