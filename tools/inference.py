import os
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import requests
from PIL import Image

os.environ["https_proxy"] = "58.34.83.134:31280"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Pix2StructForConditionalGeneration.from_pretrained('/cpfs01/shared/ADLab/hug_ckpts/structchart_ckpt')

processor = Pix2StructProcessor.from_pretrained('/cpfs01/shared/ADLab/hug_ckpts/pix2struct-chartqa-base')
processor.image_processor.is_vqa = False # Turn on if you want to use the heading text

model.to(device)

image = Image.open("/cpfs01/user/zhangbo/code/StructChart/data/test/test_chart_13.png")  #the absolute path of input image
question = "Generate a data table that corresponds to the figure depicted below."
question_qa = "Generate a data table that corresponds to the figure depicted below."

# inputs = processor(images=image, text=question, return_tensors="pt")
inputs = processor(images=image, return_tensors="pt")
inputs = inputs.to(device)

predictions = model.generate(**inputs, max_new_tokens=1280)
text = processor.decode(predictions[0], skip_special_tokens=True)
print(text)