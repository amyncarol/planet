from Dataset import KaggleAmazonDataset
from Net import KaggleAmazonNet
from torchvision import transforms
import torch.optim as optim
from Trainer import Trainer
from torch.utils.data import DataLoader

if __name__=='__main__':

	##setting environmnet variables
	NCORE = 2
	IMG_PATH = 'data/train-jpg/'
	IMG_EXT = '.jpg'
	TRAIN_DATA = 'data/train_v2.csv'

	transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

	dset_train = KaggleAmazonDataset(TRAIN_DATA, IMG_PATH, IMG_EXT, transform = transformations)

	train_loader = DataLoader(dset_train, 
							batch_size=256, 
							shuffle=True, 
							num_workers = NCORE)

	model = KaggleAmazonNet()

	optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.5)


	trainer = Trainer(optimizer)
	for epoch in range(1, 2):
		trainer.train(model, train_loader, epoch)

