import os
import csv
from glob import glob
from PIL import Image
import numpy as np
import torch
from tqdm.contrib import tzip
from transformers import Pix2StructProcessor
import random

splits = ["train", "val", "test"]
subset = ["augmented", "human"]  # "human", 
root = "${PATH_TO_DATASET_CHARTQA}"
save_root = "/cpfs01/user/dataset/chartQA_csv_heading"
os.makedirs(save_root, exist_ok=True)


input_tokenizer = Pix2StructProcessor.from_pretrained("./hug_ckpts/pix2struct-chartqa-base")
label_tokenizer = Pix2StructProcessor.from_pretrained("./hug_ckpts/pix2struct-chartqa-base")
label_tokenizer.image_processor.is_vqa = False
input_tokenizer.image_processor.is_vqa = False


for split in splits:
    
    sub_split_save_root = os.path.join(save_root, split)
    os.makedirs(sub_split_save_root, exist_ok=True)
    
    img_folder_path = os.path.join(root, split, "png")
    tables_folder_path = os.path.join(root, split, "tables")
    
    all_image_name = glob(os.path.join(img_folder_path, "*.png"))
    imgnames = []
    imgs = []
    texts = []
    for item in all_image_name:
        imgname = os.path.split(item)[-1]
        
        image = Image.open(os.path.join(img_folder_path, imgname))
        table_path = os.path.join(tables_folder_path, imgname.replace(".png", ".csv"))
        text = ""
        with open(table_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:  # Iterate through the loop to read line by line
                text = text + " \\t ".join(line) + " \\n "
                
        imgnames.append(imgname)
        imgs.append(image)
        texts.append(text)

    length = []
    for idx, (name, img, la) in enumerate(tzip(imgnames, imgs, texts)):
        # length.append(len(la))
    # print(max(length))  # 1814  1426  1143
        inputs = input_tokenizer(
                images=img,
                #text=random.choice(prompts),
                return_tensors="pt",
                padding="max_length",
                # truncation=True,
                max_patches=input_tokenizer.image_processor.max_patches,
                max_length=1280,
                )

        labels = label_tokenizer(
                text=la, 
                return_tensors="pt", 
                padding="max_length",
                # truncation=True,
                add_special_tokens=True, 
                max_length=1280,
            ).input_ids
               
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_input_flattened_patches.npy", inputs.data['flattened_patches'].numpy())
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_input_attention_mask.npy", inputs.data['attention_mask'].numpy())
        np.save(f"{sub_split_save_root}/{name.split('.')[0]}_{idx}_label.npy", labels.numpy())
        
    #     length.append(torch.sum(labels != 0))
    # print(max(length))  # 1090  852  775
            
            
        