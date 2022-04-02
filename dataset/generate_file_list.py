from pathlib import Path
from random import sample

CITYSCAPES_PATH: Path = Path("/home/cornehaasjes/data/Cityscapes/leftImg8bit/train")
CITYSCAPES_LIST_PATH: Path = Path("/home/cornehaasjes/data/Cityscapes")

GTA5_PATH: Path = Path("/home/cornehaasjes/data/GTAV/images")
GTA5_LIST_PATH: Path = Path("/home/cornehaasjes/data/GTAV")

N: int = 12500 # None #4000
LIST_OUTPUT_NAME: str = f"train-{'all' if N is None else N}.txt"
# GTA5_OUTPUT_NAME: str = f"train-{'all' if N is None else N}.txt"

cityscapes_images = [str(f.relative_to(CITYSCAPES_PATH)) for f in CITYSCAPES_PATH.glob("**/*.png")]
gta5_images = [str(f.relative_to(GTA5_PATH)) for f in GTA5_PATH.glob("**/*.png")]


def get_list_sample(image_list: "list[str]", k=None):
    if k is None:
        return sorted(image_list)
    else:
        if k > len(image_list):
            image_list *= k // len(image_list) + 1
        
        return sorted(sample(image_list, k=k))


def write_image_list(image_list: "list[str]", path: Path):
    with open(path, "w+") as f:
        f.writelines(line + "\n" for line in image_list)


write_image_list(get_list_sample(cityscapes_images, k=N), CITYSCAPES_LIST_PATH / LIST_OUTPUT_NAME)
write_image_list(get_list_sample(gta5_images, k=N), GTA5_LIST_PATH / LIST_OUTPUT_NAME)
