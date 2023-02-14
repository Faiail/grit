from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ArtGraphDataset(Dataset):
    def __init__(self, data: pd.DataFrame, root: str, transform):
        self.data = data
        self.transform = transform
        self.root = root

    def __getitem__(self, item):
        raw = self.data.iloc[item]
        img, captions = raw['name'], raw['captions']
        img = Image.open(f'{self.root}/{img}').convert('RGB')
        img = self.transform(img)
        return img, captions

    def __len__(self):
        return len(self.data)

