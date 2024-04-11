import csv
import os

from PIL import Image

from p2s.datasets.base_dataset import ChartQABASE


class ChartQADeplot(ChartQABASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        data = self.json_data[idx]
        imgname = data["imgname"]
        table_path = os.path.join(self.tables_folder_path, imgname.replace(".png", ".csv"))

        image = Image.open(os.path.join(self.img_folder_path, imgname))

        encoding = self.processor(
            images=image, 
            return_tensors="pt",
            max_patches=self.max_patches
            )  # flattened_patches, attention_mask

        encoding = {k: v.squeeze() for k, v in encoding.items()}

        text = ""
        with open(table_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:  # Iterate through the loop to read line by line
                text = text + " \\t ".join(line) + " \\n "

        encoding["labels"] = text

        return encoding

    def __len__(self) -> int:
        return len(self.json_data)


if __name__ == '__main__':
    dataset = ChartQADeplot()
    item = next(iter(dataset))
    