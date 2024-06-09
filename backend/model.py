import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path= os.path.join(current_dir, "..", "models","cnn_model.pt")

def load_data():
    """
        Load MNIST data
        
        Returns 
            loaders: Object contatining train adn test data of MNIST
    """
    train_dataloader = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_dataloader = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
)
    loaders = {
        'train': DataLoader(train_dataloader,
                            batch_size=100,
                            shuffle=True,
                            ),

        'test': DataLoader(test_dataloader,
                           batch_size=100,
                           shuffle=True,
                           )
    }
    return loaders

class CNN(nn.Module):
    """
        CNN Class

        Initial Configuration of the CNN model to be trained and tested
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  


def train_loop(dataloader, model, loss_fn, optimizer):
    #Train over the entire dataset once

    #sets the model in training mode, behaves differently during eval
    model.train()
    size = len(dataloader['train'].dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for batch, (image,label) in enumerate(dataloader['train']):
        
        pred=model(image)
        loss=loss_fn(pred, label)
        train_loss += loss_fn(pred, label).item()
        correct += (pred.argmax(1) == label).type(torch.float).sum().item()

        #backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(image)
            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')
    
    train_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

def test_loop(dataloader, model, loss_fn):
 
    #evaluation mode
    model.eval()
    size = len(dataloader['test'].dataset)
    num_batches = len(dataloader['test'])
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for image, label in dataloader['test']:
            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def make_model(n_epochs):
    model=CNN()
    loaders=load_data()
    #Hyperparameters and loss fn
    learning_rate=1e-3
    batch_size=64

    loss_fn= nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = n_epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(loaders, model, loss_fn, optimizer)
        test_loop(loaders, model, loss_fn)
    print("Done!")

    torch.save(model, model_path)


def predict_image(image):
    """
        Returns the recognized digit from the cnn model from the Input image
        Parameters
            image: Image of type PIL

        Returns: 
            pred_y: predicted value from CNN
    """
    # Load cnn model
    model = torch.load(model_path)
    model.eval()
    transform_img = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor(),])
    image_tensor = transform_img(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    with torch.no_grad():
        pred = model(image_tensor)
    predicted_class = pred.argmax(dim=1).item()
    return predicted_class

