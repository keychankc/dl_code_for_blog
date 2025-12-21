import torch, numpy as np
from collections import Counter
def box_iou_xyxy(a, b):
    # --------- 统一类型 ---------
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)

    a = a.float()
    b = b.float()

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9

    return (inter / union).float()

def error_profile(gt_boxes, pred_boxes, iou_thr=0.5):
    if len(gt_boxes)==0: return {'tp':0,'fp':len(pred_boxes),'fn':0,'dup':0}
    if len(pred_boxes)==0: return {'tp':0,'fp':0,'fn':len(gt_boxes),'dup':0}
    if isinstance(gt_boxes,list): gt_boxes=torch.tensor(gt_boxes)
    if isinstance(pred_boxes,list): pred_boxes=torch.tensor(pred_boxes)
    iou=box_iou_xyxy(gt_boxes, pred_boxes)
    gt_best=iou.max(dim=1).values
    gt_hit=gt_best>=iou_thr
    tp=int(gt_hit.sum())
    fn=int((~gt_hit).sum())
    pred_best_iou,pred_best_gt=iou.max(dim=0)
    pred_hit=pred_best_iou>=iou_thr
    fp=int((~pred_hit).sum())
    dup=0
    if pred_hit.any():
        cnt=Counter(pred_best_gt[pred_hit].tolist())
        dup=sum(v-1 for v in cnt.values() if v>1)
    return {'tp':tp,'fp':fp,'fn':fn,'dup':dup}
