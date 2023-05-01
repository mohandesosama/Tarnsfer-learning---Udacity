# Starter code for Part 1 of the Small Data Solutions Project
# 

# Set up image data for train and test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms 
from TrainModel import train_model
from TestModel import test_model
from torchvision import models


# use this mean and sd from torchvision transform documentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Set up Transforms (train, val, and test)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# Set up DataLoaders (train, val, and test)

data_dir = 'data/'
batch_size = 10
num_workers = 4

image_datasets = {x: datasets.ImageFolder(data_dir + x, data_transforms[x])
                  for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
              for x in ['train', 'val', 'test']}

# Using the VGG16 model for transfer learning 
# 1. Get trained model weights
# 2. Freeze layers so they won't all be trained again with our data
# 3. Replace top layer classifier with a classifier for our 3 categories

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 3)

# Train model with these hyperparameters
# 1. num_epochs 
# 2. criterion 
# 3. optimizer 
# 4. train_lr_scheduler 

num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# When you have all the parameters in place, uncomment these to use the functions imported above
# def main():
#     trained_model = train_model(model, criterion, optimizer, train_lr_scheduler, dataloaders['train'], dataloaders['val'], num_epochs=num_epochs)
#     test_model(dataloaders['test'], trained_model, class_names)

# if __name__ == '__main__':
#     main()
#     print("done")
