import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
class DetrWrapper:
    def __init__(self, device='cpu', score_thr=0.5):
        self.device = torch.device(device)
        self.score_thr = score_thr
        self.processor = AutoImageProcessor.from_pretrained('SenseTime/deformable-detr')
        self.model = DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr').to(self.device).eval()
    @torch.no_grad()
    def predict(self, img):
        if isinstance(img, np.ndarray): img = Image.fromarray(img)
        inputs = self.processor(images=img, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        h,w = img.size[1], img.size[0]
        res = self.processor.post_process_object_detection(outputs, target_sizes=torch.tensor([[h,w]],device=self.device), threshold=self.score_thr)[0]
        return res['boxes'].cpu().numpy().astype(np.float32), res['scores'].cpu().numpy().astype(np.float32)
