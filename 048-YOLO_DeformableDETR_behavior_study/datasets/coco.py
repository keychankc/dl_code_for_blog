import random
from pycocotools.coco import COCO
import cv2, os
class CocoDataset:
    def __init__(self, img_dir, ann_path, sample_n=300):
        self.coco = COCO(ann_path)
        self.img_dir = img_dir
        self.img_ids = random.sample(self.coco.getImgIds(), sample_n)

    def __len__(self): return len(self.img_ids)

    def __iter__(self):
        for iid in self.img_ids:
            info = self.coco.loadImgs(iid)[0]
            img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, info['file_name'])), cv2.COLOR_BGR2RGB)
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid))
            boxes = []
            for a in anns:
                x,y,w,h = a['bbox']
                boxes.append([x,y,x+w,y+h])
            yield img, boxes
