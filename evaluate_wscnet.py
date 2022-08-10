import torch
import torch.nn as nn
import sys
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from models.wscnet_simclr import WSCNetSimCLR
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

train_dataset = datasets.ImageFolder(os.path.join('./../emotion_dataset', 'train'), 
                                                    transforms.Compose([transforms.Resize((448,448)),
                                                                        transforms.ToTensor()]))

train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8)

test_dataset = datasets.ImageFolder(os.path.join('./../emotion_dataset', 'val'), 
                                                    transforms.Compose([transforms.Resize((448,448)),
                                                                        transforms.ToTensor()]))

test_loader = DataLoader(test_dataset, batch_size=32, num_workers=8)

# model = models.resnet50(pretrained=False, num_classes=8).to(device)

# checkpoint = torch.load('checkpoint_0030.pth.tar', map_location=device)
# state_dict = checkpoint['state_dict']


model = WSCNetSimCLR(1000).to(device)
model.backbone.classifier = nn.Sequential(nn.Linear(4096, 1024), nn.Linear(1024, 8)).to(device)

checkpoint = torch.load('wscnet_checkpoint_0002.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']

for k in list(state_dict.keys()):
  if k.startswith('backbone.classifier'):  
    del state_dict[k]

log = model.load_state_dict(state_dict, strict=False)
assert log.missing_keys == ['backbone.classifier.0.weight', 'backbone.classifier.0.bias', 'backbone.classifier.1.weight', 'backbone.classifier.1.bias']


# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['backbone.classifier.0.weight', 'backbone.classifier.0.bias', 'backbone.classifier.1.weight', 'backbone.classifier.1.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 4  # fc.weight, fc.bias

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 30
for epoch in range(epochs):
  top1_train_accuracy = 0
  for counter, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    top1 = accuracy(logits, y_batch, topk=(1,))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(model.backbone.classifier[0].bias)

  top1_train_accuracy /= (counter + 1)

  top1_accuracy = 0
  for counter, (x_batch, y_batch) in tqdm(enumerate(test_loader)):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
  
    top1 = accuracy(logits, y_batch, topk=(1,))
    top1_accuracy += top1[0]
  
  top1_accuracy /= (counter + 1)
  
  print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}")