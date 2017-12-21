##Mamy Ratsimbazafy - Starting Kit for PyTorch Deep Learning

import pandas as pd 
from torch import np

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset  
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import MultiLabelBinarizer

class KaggleAmazonDataset(Dataset):
	"""
	Dataset wrapping images and target labels for Kaggle - Planet Amazon competiton
	args:
	a csv file path
	path to image folder
	extension of images
	PIL transforms
	"""

	def __init__(self, csv_path, img_path, img_ext, transform=None):
		tmp_df = pd.read_csv(csv_path)
		assert tmp_df['image_name'].apply(lambda x: os.path.isfile(img_path + x + img_ext)).all(), \
		"Some images in CSV not found in the folder"

		self.mlb = MultiLabelBinarizer()
		self.img_path = img_path
		self.img_ext = img_ext
		self.transform = transform

		self.X_train = tmp_df['image_name']
		self.y_train = self.mlb.fit_transform(tmp_df['tags'].str.split()).astype(np.float32)

	def __getitem__(self, index):
		img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
		img = img.convert('RGB')
		if self.transform is not None:
			img = self.transform(img)
		label = torch.from_numpy(self.y_train[index])
		return img, label
	def __len__(self):
		return len(self.X_train.index)


