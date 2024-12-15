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
        
    def coco_to_yolo(self):
        os.makedirs(self.labels_dir, exist_ok=True)
        curr_label_dir = os.path.join(self.labels_dir, self.split)
        os.makedirs(curr_label_dir, exist_ok=False)
        all_labels = self.coco.loadAnns(self.coco.getAnnIds())
        all_imgs = self.coco.loadImgs(self.coco.getImgIds())
        
        for label, img_data in tqdm(zip(all_labels, all_imgs), desc="Converting COCO labels to YOLO format", total=len(all_labels)):
            bbox = label['bbox']
            class_label = label['category_id']
            width = img_data['width']
            height = img_data['height']
            x = bbox[0]/width
            y = bbox[1]/height
            w = bbox[2]/width
            h = bbox[3]/height
            
            final_label = str(class_label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n"
            img_label_name = img_data["file_name"].replace(".jpg", ".txt")
            with open(os.path.join(curr_label_dir, img_label_name), "a") as file:
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
