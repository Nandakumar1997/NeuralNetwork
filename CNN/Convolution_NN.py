import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
load_train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
load_test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_data = DataLoader(dataset=load_train_data, batch_size=32, shuffle=True)
test_data = DataLoader(dataset=load_test_data, batch_size=32, shuffle=False)

# Hyperparameters
n_epochs = 100
learning_rate = 0.0001


# Define model
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

model = Con_NeuralNet().to(device)

# Loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For plots during training
plt.ion()
losses = []
steps = []

# Training loop
for epoch in range(n_epochs):
    for step, (images, targets) in enumerate(train_data):
       
       
        #images = images.view(-1, 784).to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        

        if (step + 1) % 100 == 0:
            global_step = epoch * len(train_data) + step
            losses.append(loss.item())
            steps.append(global_step)
            print(f"Epoch [{epoch+1}/{n_epochs}], Step [{step+1}/{len(train_data)}], Loss: {loss.item():.4f}")
            #clear_output(wait=True)
            plt.clf()
            plt.title("Training Loss")
            plt.plot(steps, losses)
            plt.xlabel("Training Step")  
            plt.ylabel("Loss")
            plt.pause(0.01)

plt.ioff()  
plt.show()

# Evaluate on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, targets in test_data:
        
        targets = targets.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "cifar_model.pth")
