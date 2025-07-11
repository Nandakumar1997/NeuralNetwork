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
load_train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
load_test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

train_data = DataLoader(dataset=load_train_data, batch_size=32, shuffle=True)
test_data = DataLoader(dataset=load_test_data, batch_size=32, shuffle=False)

# Hyperparameters
n_epochs = 500
learning_rate = 0.0001
input_size = 784
first_hidden_size = 512
seconnd_hidden_size=256
output_size = 10

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_layer, hidden_size_1,hidden_size_2, output):
        super().__init__()
        self.linear_1 = nn.Linear(input_layer, hidden_size_1)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu_2=nn.ReLU()
        self.linear_3=nn.Linear(hidden_size_2,output)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu_1(out)
        out = self.linear_2(out)
        out=self.relu_2(out)
        out=self.linear_3(out)
        return out

model = NeuralNet(input_size, first_hidden_size,seconnd_hidden_size, output_size).to(device)

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
        images = images.view(-1, 784).to(device)
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
        images = images.view(-1, 784).to(device)
        targets = targets.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

accuracy = (correct / total) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "mnist_model.pth")
