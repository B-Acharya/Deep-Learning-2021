import torch
from tqdm import tqdm
# from src.models.variational_encoder import get_digitis
##########################################
#       generic train function           #
##########################################


def train (model, dataloader, dataset, device, optimizer, loss):

	model.train()
	running_loss = 0
	counter = 0

	for i , data in tqdm(enumerate(dataloader), total= int(len(dataset/dataloader.batch_size))):
		counter += 1
		data - data[0]
		pass

def validate(model, dataloader, device, output_images,  criterion):
	model.eval()
	running_loss = 0
	counter =0
	with torch.no_grad():
		for data, label in tqdm(dataloader):
			counter += 1
			data = data.to(device)
			x_pred = model(data)
			# loss = criterion(x_pred, data) + model.encoder.kl
			loss = criterion(x_pred, output_images[label].unsqueeze(0).to(device)) + model.encoder.kl
			running_loss += loss.item()
		return running_loss/counter
