import numpy as np
import torch
from ultralytics import YOLO

class YOLOWrapper:
    def __init__(self, device="cpu", score_thr=0.25, model_name="yolov8n.pt", imgsz=640):
        self.device = device
        self.score_thr = float(score_thr)
        self.imgsz = int(imgsz)

        # device: "cpu" / "cuda" / "cuda:0"
        self.model = YOLO(model_name)
        try:
            self.model.to(device)
        except Exception:
            # Ultralytics 在部分版本里 to(device) 可能不可用，但 predict 会吃 device 参数
            pass

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray):
        """
        img_rgb: numpy RGB (H,W,3)
        returns: boxes_xyxy (N,4), scores (N,)
        """
        # Ultralytics 接受 numpy RGB
        results = self.model.predict(
            source=img_rgb,
            imgsz=self.imgsz,
            conf=self.score_thr,
            device=self.device,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        boxes = results.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.detach().cpu().numpy().astype(np.float32)

        # 再保险过滤一次（不同版本 conf 行为略有差异）
        keep = scores >= self.score_thr
        return boxes[keep], scores[keep]