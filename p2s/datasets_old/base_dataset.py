import json
import os

from torch.utils.data import Dataset
from transformers import Pix2StructProcessor


class ChartQABASE(Dataset):
    def __init__(
            self,
            root: str = "ChartQA_Dataset",
            split: str = "train",
            subset: str = "W" ,  # W: whole    H: human    M: augmented
            max_patches: int = 4096,  # Base: 4096  Large: 3072
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        assert subset in ["W", "H", "M"]
        
        self.subset_name = {
            "W": "whole",
            "H": "human",
            "M": "augmented"
        }
        
        self.root = root
        self.split = split
        self.max_patches = max_patches
        self.subset = subset

        self.json_augmented_file_path = os.path.join(self.root, self.split, f"{split}_augmented.json")
        self.json_human_file_path = os.path.join(self.root, self.split, f"{split}_human.json")

        self.all_json_data = []
        with open(self.json_augmented_file_path, 'r') as f:
            self.json_augmented_data = json.load(f)
            self.all_json_data.extend(self.json_augmented_data)

        with open(self.json_human_file_path, 'r') as f:
            self.json_human_data = json.load(f)
            self.all_json_data.extend(self.json_human_data)

        self.annotations_folder_path = os.path.join(self.root, self.split, "annotations")
        self.img_folder_path = os.path.join(self.root, self.split, "png")
        self.tables_folder_path = os.path.join(self.root, self.split, "tables")
        
        if self.subset == "W":
            self.json_data = self.all_json_data
        elif self.subset == "H":
            self.json_data = self.json_human_data
        elif self.subset == "M":
            self.json_data = self.json_augmented_data
        # print('Done')

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses should implement this!")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses should implement this!")
