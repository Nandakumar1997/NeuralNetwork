import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, ToPILImage

# Load MNIST test set
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())

# Extract a sample image of the digit "3"
for img, label in test_dataset:
    if label == 1:
        pil_img = ToPILImage()
        pil_img=pil_img(img)
        pil_img.save("mnist.png")
        pil_img.show()
        break
