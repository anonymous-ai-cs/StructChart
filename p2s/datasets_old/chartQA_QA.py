import os

from PIL import Image

from p2s.datasets.base_dataset import ChartQABASE


class ChartQA(ChartQABASE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int):
        data = self.json_data[idx]
        imgname = data["imgname"]
        query = data["query"]
        label = data["label"]

        image = Image.open(os.path.join(self.img_folder_path, imgname))
 
        return image, query, label

    def __len__(self) -> int:
        return len(self.json_data)


if __name__ == '__main__':
    dataset = ChartQA()
    item = next(iter(dataset))
