from torch.utils.data import DataLoader
from vision.coco_dataset import COCODataset




class COCOTrainer():
    def __init__(self, dataset: COCODataset, batch_size: int = 8, model = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=dataset.collate_fn())
    
    def train(self):
        # Iterating through the dataloader
        for images, bboxes, labels in self.loader:
            print("Images batch shape:", len(images))
            print("Bounding boxes batch shape:", len(bboxes))
            print("Labels batch shape:", len(labels))
            break