import torch
from utils import globals


def print_torch_info():
    print(f"{globals.TAGS['INFO_TAG']} PyTorch version: {torch.__version__}")
    
    print(f"{globals.TAGS['INFO_TAG']} CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{globals.TAGS['INFO_TAG']} CUDA version: {torch.version.cuda}")
        print(f"{globals.TAGS['INFO_TAG']} cuDNN version: {torch.backends.cudnn.version()}")
        print(f"{globals.TAGS['INFO_TAG']} Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"{globals.TAGS['INFO_TAG']}     GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{globals.TAGS['INFO_TAG']} Default device: {device}")
    
    # Compilation details
    print(f"{globals.TAGS['INFO_TAG']} Compiled with CUDA: {torch.version.cuda is not None}")
    print(f"{globals.TAGS['INFO_TAG']} Debug build: {torch.version.debug}")
    
    print(f"{globals.TAGS['INFO_TAG']} Available Torch libraries and extensions:")
    available_torch_libraries = ["torchvision", "torchaudio", "torchtext", "torchserve"]
    for lib in available_torch_libraries:
        try:
            __import__(lib)
            print(f"{globals.TAGS['INFO_TAG']}   - {lib}: Installed")
        except ImportError:
            print(f"{globals.TAGS['INFO_TAG']}   - {lib}: Not installed")

def create_yolo_splits(root, split="train2017"):
    path = os.path.join(root, "images", split)
    all_imgs = os.listdir(path)
    
    
    with open(os.path.join(root, split+".txt"), "w") as file:
        for img_name in all_imgs:
            if img_name != all_imgs[-1]:
                file.write(os.path.join(path, img_name)+"\n")
            else:
                file.write(os.path.join(path, img_name))
