
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas as pd
import matplotlib.pyplot as plt

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):

    def __init__(self, data, mode: str):

        super().__init__()
        self.data = data
        self.mode = mode
        if self.mode == "train":
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(degrees=20),
            ])
        else:
            self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std),
            ])

    def __len__(self):  # 返回data的行数
        return self.data.shape[0]

    def __getitem__(self, index):

        # if torch.is_tensor(index):
        #     index = index.tolist()

        # image_path = self.data.iloc[index, 0]
        image = imread(Path(self.data.iloc[index, 0]))
        # 转换成三通道的图
        image = gray2rgb(image)
        # image
        labels = self.data.iloc[index, 1:]

        labels = torch.tensor((self.data.iloc[index]['crack'], self.data.iloc[index]['inactive']),
                              dtype=torch.float32)
        if self._transform:
            image = self._transform(image)
            # image = image.unsqueeze(0)


        return image, labels









# from torch.utils.data import Dataset
# import torch
# from pathlib import Path
# from skimage.io import imread
# from skimage.color import gray2rgb
# import numpy as np
# import torchvision as tv
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# train_mean = [0.59685254, 0.59685254, 0.59685254]
# train_std = [0.16043035, 0.16043035, 0.16043035]
#
#
# class ChallengeDataset(Dataset):
#
#     def __init__(self, data, mode: str):
#
#         super().__init__()
#         self.data = data
#         self.mode = mode
#         if self.mode == "train":
#             self._transform = tv.transforms.Compose([
#                 tv.transforms.ToPILImage(),
#                 tv.transforms.RandomHorizontalFlip(),
#                 tv.transforms.ToTensor(),
#                 tv.transforms.Normalize(mean=train_mean, std=train_std),
#                 tv.transforms.RandomHorizontalFlip(),
#                 tv.transforms.RandomRotation(degrees=20),
#             ])
#         else:
#             self._transform = tv.transforms.Compose([
#                 tv.transforms.ToPILImage(),
#                 tv.transforms.ToTensor(),
#                 tv.transforms.Normalize(mean=train_mean, std=train_std),
#             ])
#
#     def __len__(self):  # 返回data的行数
#         return self.data.shape[0]
#
#     def __getitem__(self, index):
#
#         # if torch.is_tensor(index):
#         #     index = index.tolist()
#
#         # image_path = self.data.iloc[index, 0]
#         image = imread(Path(self.data.iloc[index, 0]))
#         # 转换成三通道的图
#         image = gray2rgb(image)
#         # image
#         labels = self.data.iloc[index, 1:]
#
#         labels = torch.tensor((self.data.iloc[index]['crack'], self.data.iloc[index]['inactive']),
#                               dtype=torch.float32)
#         if self._transform:
#             image = self._transform(image)
#             # image = image.unsqueeze(0)
#
#
#         return image, labels