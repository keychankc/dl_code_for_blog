import argparse
from tqdm import tqdm
from datasets.coco import CocoDataset
from wrappers.yolo_wrapper import YOLOWrapper
from wrappers.detr_wrapper import DetrWrapper
from utils.boxes import error_profile

def summarize(err_list):
    out = {"tp": 0, "fp": 0, "fn": 0, "dup": 0}
    for e in err_list:
        for k in out:
            out[k] += e[k]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', required=True)
    ap.add_argument('--ann_path', required=True)
    ap.add_argument('--sample_n', type=int, default=300)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    ds = CocoDataset(args.img_dir, args.ann_path, sample_n=args.sample_n)
    yolo = YOLOWrapper(device=args.device)
    detr = DetrWrapper(device=args.device)

    summary = {'yolo': [], 'detr': []}
    for img, gt_boxes in tqdm(ds):
        yb, _ = yolo.predict(img)
        db, _ = detr.predict(img)
        summary['yolo'].append(error_profile(gt_boxes, yb))
        summary['detr'].append(error_profile(gt_boxes, db))

    print("=== Error Summary ===")
    print("all samples:", len(summary["yolo"]))
    print(" YOLO:", summarize(summary["yolo"]))
    print(" DETR:", summarize(summary["detr"]))

if __name__ == "__main__":
    main()
