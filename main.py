from vision.coco_dataset import COCODataset
from vision.coco_train import COCOTrainer
import argparse
from utils import globals
from utils import utils

import os
os.system("color")

utils.print_torch_info()



def main(args):
    coco = COCODataset(args.coco_root, split="train")
    coco.show_sample()
    trainer = COCOTrainer(coco)
    trainer.train()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry Point of VQA Algo")
    parser.add_argument("--coco_root", type=str, help=globals.HELP_MSGS["COCO_ROOT_HELP"])
    
    args = parser.parse_args()
    
    main(args)