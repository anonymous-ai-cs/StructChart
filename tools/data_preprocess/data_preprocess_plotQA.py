import os
import csv
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from tqdm.contrib import tzip
from transformers import Pix2StructProcessor
import random
import math

splits = ["train", "val"]
root = "${PATH_TO_DATASET_PLOTQA}"
save_root = "/cpfs01/user/dataset/plotqa_csv_0.01"
os.makedirs(save_root, exist_ok=True)


tokenizer = Pix2StructProcessor.from_pretrained("./hug_ckpts/pix2struct-chartqa-base")
tokenizer.image_processor.is_vqa = False


for split in splits:
    
    sub_split_save_root = os.path.join(save_root, split)
    os.makedirs(sub_split_save_root, exist_ok=True)
    
    img_folder_path = os.path.join(root, split, "png")
    json_train_file_path = os.path.join(root, split, f"annotations_{split}.json")
    
    all_image_name = glob(os.path.join(img_folder_path, "*.png"))
    
    with open(json_train_file_path, 'r') as f:
            json_data = json.load(f)
    
    imgnames = []
    imgs = []
    texts = []
    if split == 'train':
        sample_size = math.ceil(len(json_data)*0.01)   
    if split == 'val':
        sample_size = math.ceil(len(json_data)*0.05)       
    sample_data = random.sample(json_data, sample_size)
    
    for item in tqdm(sample_data):
        imgname = item["image_index"]
        img_type = item["type"]
        value = item["models"]
        img = Image.open(os.path.join(img_folder_path, f"{imgname}.png"))
        lines = []
        if img_type == "hbar_categorical":
            x_label = item["general_figure_info"]["y_axis"]["major_labels"]["values"]
            x_name = item["general_figure_info"]["x_axis"]["label"]["text"]
            y_name = item["general_figure_info"]["y_axis"]["label"]["text"]
            for item1 in value:
                label = item1 ["name"]
                y = item1["x"]
                line = " \\t".join([label]+ [str(round(num,1)) for num in y])
                lines.append(line+" \\n")
            x = [str(num) for num in x_label][:(len(x_label)//2)]
            x_value = "Entity" + " / " + x_name + " / " + y_name + " \\t" + " \\t".join(x[:-1])+ " \\t" + x[-1] + " \\n"
            csv_per = "".join([x_value] + lines)


        else:
            x_label = item["general_figure_info"]["x_axis"]["major_labels"]["values"]
            x_name = item["general_figure_info"]["x_axis"]["label"]["text"]
            y_name = item["general_figure_info"]["y_axis"]["label"]["text"]
            for item1 in value:
                label = item1 ["name"]
                y = item1["y"]
                line = " \\t".join([label]+ [str(round(num,1)) for num in y])
                lines.append(line+"\\n")
            x = [str(num) for num in x_label][:(len(x_label)//2)]
            x_value = "Entity" + " / " + y_name + " / " + x_name + " \\t" + " \\t".join(x[:-1])+ " \\t" + x[-1] + " \\n"
            csv_per = "".join([x_value] + lines)
        texts.append(csv_per)
        imgnames.append(imgname)
        imgs.append(img)

    for idx, (name, img, la) in enumerate(tzip(imgnames, imgs, texts)):
        inputs = tokenizer(
                images=img,
                return_tensors="pt",
                padding="max_length",
                # truncation=True,
                max_patches=tokenizer.image_processor.max_patches,
                max_length=20,
                )

        labels = tokenizer(
                text=la, 
                return_tensors="pt", 
                padding="max_length",
                # truncation=True,
                add_special_tokens=True, 
                max_length=1280,
            ).input_ids
               
        np.save(f"{sub_split_save_root}/{name}_{idx}_plotqa_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
        np.save(f"{sub_split_save_root}/{name}_{idx}_plotqa_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
        np.save(f"{sub_split_save_root}/{name}_{idx}_plotqa_label.npy", labels.numpy())
   
    


