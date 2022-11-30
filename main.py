
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam, SGD, lr_scheduler
from torch.autograd import Variable

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

d_mean = [0.485, 0.456, 0.406]
t_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize(d_mean, t_std),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = datasets.ImageFolder('training_set_t', transform=transform)
N_CLASSES = len(dataset.classes)

train_set, val_set = random_split(dataset, [0.8, 0.2])
trainloader, valloader = DataLoader(train_set, batch_size=64, shuffle=True), DataLoader(val_set, batch_size=64)

def imageshow(img):
    #img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#images, _ = next(iter(trainloader))
#for i, image in enumerate(images[:3]):
#    imageshow(image)

class mush_rec_nn(nn.Module):
    def __init__(self):
        super(mush_rec_nn, self).__init__()
        
        k_s = 3
        pad = 0

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=k_s, stride=1, padding=pad)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(12)
        
        self.cnn2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=k_s, stride=1, padding=pad)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(24)
        
        self.cnn3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=k_s, stride=1, padding=pad)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.cnn4 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=k_s, stride=1, padding=pad)
        self.pool4 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(96)
        
        self.cnn5 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=k_s, stride=1, padding=pad)
        self.pool5 = nn.MaxPool2d(2)
        self.bn5 = nn.BatchNorm2d(192)
        
        linr = 192 * 1 * 1
        linr2 = linr
        self.fc1 = nn.Linear(linr, linr)
        self.fc2 = nn.Linear(linr, 192)
        self.dp1 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(192, N_CLASSES)
    
    def forward(self, input):
        output = self.pool1(F.relu(self.bn1(self.cnn1(input))))
        output = self.pool2(F.relu(self.bn2(self.cnn2(output))))  
        output = self.pool3(F.relu(self.bn3(self.cnn3(output))))
        output = self.pool4(F.relu(self.bn4(self.cnn4(output))))  
        output = self.pool5(F.relu(self.bn5(self.cnn5(output))))
        
        output = output.view(-1, 192 * 1 * 1)
        
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        output = self.dp1(output)
        output = self.fc3(output)

        return output

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        # Convolution 2
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.cnn5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        
        self.cnn6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU()

        self.fc1 = nn.Linear(512 *6*6, 512)
        self.relu7 = nn.ReLU()
        self.dp1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, N_CLASSES)
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        
        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        
        out = self.cnn3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.maxpool(out)
        
        out = self.cnn4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.maxpool(out)
        
        out = self.cnn5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.maxpool(out)
        
        #out = self.cnn6(out)
        #out = self.bn6(out)
        #out = self.relu6(out)
        #out = self.maxpool(out)
        

        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        out = self.relu7(out)        
        out = self.dp1(out)
        out = self.fc2(out)
        return out
    
model = mush_rec_nn()

lr = 0.1
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

def saveModel():
    path = "./mush_rec_model.pth"
    torch.save(model.state_dict(), path)

def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    accuracy = (100 * accuracy / total)
    return(accuracy)

def train(num_epochs):
    global trainloader
    
    best_accuracy = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        train_loss = []
        #train_losses = []
        
        for i, (images, labels) in enumerate(trainloader, 0):

            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            optimizer.step()

            train_loss.append(loss.item())
            running_loss += loss.item()
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        #sheduler.step(np.mean(train_loss))
        
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

        trainloader = DataLoader(train_set, batch_size=64, shuffle=True)

def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(valloader))

    # show all images as one image grid
    imageshow(make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

# Let's build our model
train(100)
print('Finished Training')
# Test which classes performed well
testAccuracy()

# Let's load the model we just created and test the accuracy per label
model = mush_rec_nn()
path = "./mush_rec_model.pth"
model.load_state_dict(torch.load(path))

"""
. . . . 
.
.
.
"""



