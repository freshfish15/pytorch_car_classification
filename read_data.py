import os
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
from PIL import Image
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "data/Cars"
black_car = "black"
white_car = "white"
black_dataset = MyData(root_dir, black_car)
white_dataset = MyData(root_dir, white_car)

train_dataset = black_dataset + white_dataset