from pycocotools.coco import COCO
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from utils import globals
import matplotlib.pyplot as plt

from tqdm import tqdm

class COCODataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.coco_root_dir = root_dir
        self.split = "train2017" if split == "train" else "val2017" if split == "val" else "test2017"
        self.img_dir = os.path.join(self.coco_root_dir, "images", self.split)
        self.train_ann_path = os.path.join(self.coco_root_dir, "annotations", f"instances_{self.split}.json")
        self.coco = COCO(self.train_ann_path)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform
        
    def __str__(self):
        split = f"{globals.TAGS['DATASET_TAG']} Current split is: {self.split}" + "\n"
        root_dir = f"{globals.TAGS['DATASET_TAG']} COCO root dir: {self.coco_root_dir}\n"
        img_dir = f"{globals.TAGS['DATASET_TAG']} Images directory is: {self.img_dir}\n"
        ann_path = f"{globals.TAGS['DATASET_TAG']} Annotation path is: {self.train_ann_path}\n"
        #ids = f"COCO ids: {self.img_ids}\n"
 
        return split + root_dir + img_dir + ann_path    
    
    def collate_fn(self):
        return lambda x: tuple(zip(*x))
        
    @staticmethod
    def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
        box_width = bbox[2]
        box_height = bbox[3]
        
        x = (bbox[0] + box_width / 2) / img_width
        y = (bbox[1] + box_height / 2) / img_height
        w = box_width / img_width
        h = box_height / img_height
        
        return x, y, w, h
    
    @staticmethod
    def get_category_name(all_categories, id):
        for cat in all_categories:
            if cat["id"] == id:
                return cat["name"]
    
    def coco_to_yolo(self):
        os.makedirs(self.labels_dir, exist_ok=True)
        curr_label_dir = os.path.join(self.labels_dir, self.split)
        os.makedirs(curr_label_dir, exist_ok=True)
        
        all_labels = self.coco.loadAnns(self.coco.getAnnIds())
        all_imgs = self.coco.loadImgs(self.coco.getImgIds())
        all_categories = self.coco.loadCats(self.coco.getCatIds())
        ann_ids = self.coco.getAnnIds(imgIds=all_imgs[0]["id"])
        anns = self.coco.loadAnns(ann_ids)
        
        
        
        for img_data in  tqdm(all_imgs, desc="Converting COCO labels to YOLO format", total=len(all_imgs)):
            ann_ids = self.coco.getAnnIds(imgIds=img_data["id"])
            anns = self.coco.loadAnns(ann_ids)
            img_width = img_data['width']
            img_height = img_data['height']
            img_label_name = img_data["file_name"].replace(".jpg", ".txt")
            final_label = ""
            
            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']
                class_name = COCODataset.get_category_name(all_categories, cat_id)
                class_idx = globals.COCO_CLASSES.index(class_name)
                x, y, w, h = COCODataset.convert_coco_bbox_to_yolo(bbox, img_width, img_height)
                
                if ann is not anns[-1]:
                    final_label += str(class_idx) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
                else:
                    final_label += str(class_idx) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h)
                    
            with open(os.path.join(curr_label_dir, img_label_name), "w") as file:
                file.write(final_label)
    
    def show_sample(self, id: int = 0):
        img_id = self.coco.getImgIds()[id]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        plt.imshow(image)
        plt.axis('off')

        # Load annotations for the image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Display annotations on the image
        self.coco.showAnns(annotations)

        plt.show()
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        # Load image info
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)
        
        # Prepare labels and bounding boxes
        bboxes = []
        labels = []
        for ann in annotations:
            bbox = ann['bbox']  # [x_min, y_min, width, height]
            bboxes.append(bbox)
            labels.append(ann['category_id'])
        
        # Convert to NumPy arrays
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        
        return image, bboxes, labels
