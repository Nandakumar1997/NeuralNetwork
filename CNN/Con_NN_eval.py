import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image, ImageOps


#hyper parameters
n_epochs = 500
learning_rate = 0.0001
input_size = 784
first_hidden_size = 512
seconnd_hidden_size=256
output_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Con_NeuralNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.conlay1 =nn.Conv2d(3,8,3)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(2,2)
        self.conlay2=nn.Conv2d(8,16,3)
        self.linear_1 = nn.Linear(16*6*6, 252)
        self.linear_2 = nn.Linear(252, 100)
        self.linear_3 = nn.Linear(100, 10)
        
    def forward(self, x):
        out=self.maxpool(self.relu(self.conlay1(x)))
        out=self.maxpool(self.relu(self.conlay2(out)))
        out = self.relu(self.linear_1(out.view(-1,16*6*6)))
        out = self.relu(self.linear_2(out))
        out = self.linear_3(out)
        return out
    
#loss calculation & optimizer
model = Con_NeuralNet().to(device)
model.load_state_dict(torch.load("cifar_model.pth"))
model.eval()



# Load and preprocess image
image = Image.open(".\\cifar.png")
#image = ImageOps.invert(image)  # Invert to match MNIST style

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                     (0.2023, 0.1994, 0.2010))

])

image = transform(image).unsqueeze(0).to(next(model.parameters()).device)


# Predict
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(dim=1).item()
    classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 
 'dog', 'frog', 'horse', 'ship', 'truck']
print(f"Predicted digit: {classes[predicted_class]}")
