#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV 鼠标点选版 SAM 交互式分割标注
- 左键：前景点（fg）
- 右键：背景点（bg）
- 键盘 p：预测（predict）
- 键盘 r：重置点（reset）
- 键盘 s：保存当前最佳掩码并进入下一张图（save）
- 键盘 q / ESC：退出（quit）
- 自动记录 clicks/time/IoU/Dice（若提供 GT）

依赖：
pip install torch torchvision opencv-python numpy pandas segment-anything

运行：
python scripts/sam_interactive_annotate.py \
  --img_dir data/images \
  --gt_dir data/gts \
  --out_dir outputs \
  --sam_ckpt checkpoints/sam_vit_h_4b8939.pth \
  --model_type vit_h \
"""
import os
import time
import json
import argparse
"""
说明：
- GT 掩码：data/gts/{image_stem}.png（0/255 或 0/1 或任意>0视为前景）
- 输出：
  outputs/maskrs/{image_stem}.png
  outputs/summary.csv
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor


# -------------------------
# Metrics
# -------------------------
def _binarize(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return None
    if mask.dtype == bool:
        return mask.astype(np.uint8)
    if mask.max() > 1:
        return (mask > 127).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def iou_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    p = _binarize(pred_mask)
    g = _binarize(gt_mask)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    iou = inter / (union + 1e-9)
    dice = (2 * inter) / (int(p.sum()) + int(g.sum()) + 1e-9)
    return float(iou), float(dice)


# -------------------------
# Logging
# -------------------------
@dataclass
class Record:
    image: str
    clicks_total: int
    pos_points: int
    neg_points: int
    elapsed_sec: float
    sam_best_score: float
    iou: Optional[float]
    dice: Optional[float]
    mask_path: str
    prompts_json: str


# -------------------------
# OpenCV UI helper
# -------------------------
class ClickUI:
    def __init__(self, win_name: str):
        self.win = win_name
        self.points: List[Tuple[int, int]] = []
        self.labels: List[int] = []  # 1 fg, 0 bg

        self._base = None      # BGR
        self._canvas = None    # BGR (draw points + overlay)
        self._overlay = None   # RGBA-like in BGR (we'll alpha blend)
        self._last_key = -1

        self.best_mask = None  # uint8 0/1
        self.best_score = -1.0

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._on_mouse)

    def set_image(self, img_bgr: np.ndarray):
        self._base = img_bgr.copy()
        self.best_mask = None
        self.best_score = -1.0
        self.points.clear()
        self.labels.clear()
        self._redraw()

    def _on_mouse(self, event, x, y, flags, param):
        # left = fg, right = bg
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(1)
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append((x, y))
            self.labels.append(0)
            self._redraw()

    def reset_points(self):
        self.points.clear()
        self.labels.clear()
        self.best_mask = None
        self.best_score = -1.0
        self._redraw()

    def set_mask(self, mask01: np.ndarray, score: float):
        self.best_mask = mask01.astype(np.uint8)
        self.best_score = float(score)
        self._redraw()

    def _redraw(self):
        if self._base is None:
            return
        canvas = self._base.copy()

        # mask overlay (blue-ish by default without explicitly choosing colors would be ideal,
        # but OpenCV needs a color; keep subtle and consistent)
        if self.best_mask is not None:
            m = self.best_mask
            overlay = canvas.copy()
            overlay[m == 1] = (overlay[m == 1] * 0.4 + np.array([255, 144, 30]) * 0.6).astype(np.uint8)
            canvas = overlay

        # points
        for (px, py), lab in zip(self.points, self.labels):
            if lab == 1:
                col = (0, 255, 0)   # green for fg
            else:
                col = (0, 0, 255)   # red for bg
            cv2.circle(canvas, (px, py), 5, col, -1)
            cv2.circle(canvas, (px, py), 10, col, 2)

        # HUD text
        pos = sum(1 for l in self.labels if l == 1)
        neg = sum(1 for l in self.labels if l == 0)
        hud1 = f"FG(LMB): {pos}  BG(RMB): {neg}  Total: {len(self.points)}"
        hud2 = f"Keys: p=predict  s=save  r=reset  q/esc=quit   best_score={self.best_score:.4f}"
        cv2.putText(canvas, hud1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, hud2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)

        self._canvas = canvas
        cv2.imshow(self.win, self._canvas)

    def wait_key(self, delay=30) -> int:
        k = cv2.waitKey(delay) & 0xFF
        return k


# -------------------------
# Main
# -------------------------
def load_gt(gt_path: str) -> np.ndarray:
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(gt_path)
    return (gt > 127).astype(np.uint8)


def save_mask(out_path: str, mask01: np.ndarray):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, (mask01.astype(np.uint8) * 255))


def build_predictor(ckpt: str, model_type: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    sam.to(device=device)
    return SamPredictor(sam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--gt_dir", default=None, help="Optional. If provided, compute IoU/Dice with GT mask.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    masks_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    if args.max_images and args.max_images > 0:
        img_files = img_files[: args.max_images]

    predictor = build_predictor(args.sam_ckpt, args.model_type, args.device)
    ui = ClickUI("SAM Annotator (OpenCV)")

    records: List[Record] = []
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    summary_json = os.path.join(args.out_dir, "summary.json")

    for idx, fname in enumerate(img_files, 1):
        img_path = os.path.join(args.img_dir, fname)
        stem = os.path.splitext(fname)[0]

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        # set SAM image (RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        ui.set_image(img_bgr)

        print(f"\n=== [{idx}/{len(img_files)}] {fname} ===")
        print("操作：鼠标左键前景、右键背景；按 p 预测；按 s 保存进入下一张；按 r 重置；按 q 退出。")

        t0 = time.time()
        best_mask = None
        best_score = -1.0

        while True:
            k = ui.wait_key(30)

            if k in (ord("q"), 27):  # q or ESC
                print("[QUIT] user quit.")
                cv2.destroyAllWindows()
                # persist progress
                if records:
                    pd.DataFrame([asdict(r) for r in records]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
                    with open(summary_json, "w", encoding="utf-8") as f:
                        json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)
                return

            if k == ord("r"):
                ui.reset_points()
                best_mask = None
                best_score = -1.0
                continue

            if k == ord("p"):
                if len(ui.points) == 0:
                    print("[WARN] 还没有点提示。请至少左键点一个前景点。")
                    continue

                point_coords = np.asarray(ui.points, dtype=np.float32).reshape(-1, 2)
                point_labels = np.asarray(ui.labels, dtype=np.int32).reshape(-1)

                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True,
                )
                bi = int(np.argmax(scores))
                best_mask = masks[bi].astype(np.uint8)
                best_score = float(scores[bi])
                ui.set_mask(best_mask, best_score)
                print(f"[PRED] candidates={len(scores)} best_score={best_score:.4f} clicks={len(ui.points)}")

            if k == ord("s"):
                if best_mask is None:
                    print("[WARN] 还没有预测结果。请先按 p 预测，再按 s 保存。")
                    continue

                elapsed = time.time() - t0
                pos = sum(1 for l in ui.labels if l == 1)
                neg = sum(1 for l in ui.labels if l == 0)

                out_mask_path = os.path.join(masks_dir, f"{stem}.png")
                save_mask(out_mask_path, best_mask)

                iou = dice = None
                if args.gt_dir:
                    gt_path = os.path.join(args.gt_dir, f"{stem}.png")
                    if os.path.exists(gt_path):
                        gt = load_gt(gt_path)
                        iou, dice = iou_dice(best_mask, gt)
                        print(f"[EVAL] IoU={iou:.4f} Dice={dice:.4f}")
                    else:
                        print(f"[EVAL] GT not found: {gt_path}")

                prompts = {
                    "points": ui.points,
                    "labels": ui.labels,
                    "label_meaning": "1=foreground(LMB), 0=background(RMB)",
                    "keys": "p=predict, s=save, r=reset, q/esc=quit"
                }

                rec = Record(
                    image=fname,
                    clicks_total=len(ui.points),
                    pos_points=pos,
                    neg_points=neg,
                    elapsed_sec=float(elapsed),
                    sam_best_score=float(best_score),
                    iou=iou,
                    dice=dice,
                    mask_path=os.path.relpath(out_mask_path, args.out_dir),
                    prompts_json=json.dumps(prompts, ensure_ascii=False),
                )
                records.append(rec)

                # persist after each save
                pd.DataFrame([asdict(r) for r in records]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
                with open(summary_json, "w", encoding="utf-8") as f:
                    json.dump([asdict(r) for r in records], f, ensure_ascii=False, indent=2)

                print(f"[SAVE] mask -> {out_mask_path}")
                print(f"[SAVE] summary -> {summary_csv}")
                break  # next image

    cv2.destroyAllWindows()
    print("\nDone. Results:")
    print(f"- Masks:   {os.path.join(args.out_dir, 'masks')}")
    print(f"- Summary: {summary_csv}")


if __name__ == "__main__":
    main()