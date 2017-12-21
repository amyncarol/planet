from torch.autograd import Variable
import torch.nn.functional as F 
class Trainer():
	def __init__(self, optimizer):
		self.optimizer = optimizer

	def train(self, model, train_loader, epoch):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data), Variable(target)
			self.optimizer.zero_grad()
			output = model(data)
			loss = F.binary_cross_entropy(output, target)
			loss.backward()
			self.optimizer.step()
			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset), 
					100. * batch_idx/len(train_loader), loss.data[0]))

