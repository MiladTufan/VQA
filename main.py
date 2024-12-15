from vision.coco_dataset import COCODataset
from vision.coco_train import COCOTrainer
import argparse
from utils import globals
from utils import utils
import config

import os
os.system("color")

utils.print_torch_info()


#########################################################################
#
# Sets default configurations for command-line arguments to reduce the 
# need for specifying many arguments manually.
#
#########################################################################
def set_config(args):
    user_args = config.PATHS[args.usr]
    cmd_args = vars(args)
    
    for key, val in user_args.items():
        cmd_args[key] = val

    args = argparse.Namespace(**cmd_args)
    return args


#########################################################################
#
# Sets default configurations for command-line arguments to reduce the 
# need for specifying many arguments manually.
#
#########################################################################
def main(args):
    coco = COCODataset(args.coco_root, split="train")
    trainer = COCOTrainer(coco)
    trainer.train()


#########################################################################
#
# Main entry point of the system.
#
#########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entry Point of VQA Algo")
    parser.add_argument("--usr", type=str, default="win", help="set default user")
    parser.add_argument("--coco_root", type=str, help="path/to/coco/root/dir")
    parser.add_argument("--train", type=str, help="Start training loop")
    
    args = parser.parse_args()
    args = set_config(args)
    main(args)