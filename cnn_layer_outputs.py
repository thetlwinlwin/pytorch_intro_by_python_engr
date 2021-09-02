import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 0
batch_size = 4
learning_rate = 0.001

transforming = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./dataset',train=True,download=True,transform=transforming)
test_dataset = torchvision.datasets.CIFAR10(root='./dataset',train=False,download=True,transform=transforming)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(img):
    img = img / 2+ 0.5 #un-normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
dataiter = iter(train_loader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)

# batch_size is 4, rgb color channel(3color),height,width
print(images.shape) # [4,3,32,32]


x = conv1(images)

# 4 batch, 6 output channels, height, width
# height and width becomes smaller.
# 32x32 is original H and W.
# filter size AKA kernel_size is 5
# padding is zero and stride is 1.
# (width - pooling size + 2 X padding)/stride + 1
# (32-5 + (2x0))/1 + 1 = 28
print(f'first conv shape {x.shape}')  #[4,6,28,28]

x = pool(x)

# 28x28 size would be reduced down to half as the pooling layer is 2x2
# 28/2 = 14   => 14x14
print(f'first pool shape {x.shape}')  # [4,6,14,14]

x = conv2(x)

# (14-5 + (2x0))/ 1 + 1 = 10
# second conv layer has 16 output channels.
print(f'second conv shape {x.shape}') # [4,16,10,10]

x = pool(x)

# 10/2 = 5
print(f'second pool shape {x.shape}') # [4,16,5,5]
