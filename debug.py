import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Classifier(ImageClassificationBase):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # 3, 128, 128
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 16, 128, 128
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 16, 64, 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 32, 64, 64
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 32, 32, 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 64, 32, 32
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 64, 16, 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 128, 16, 16
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 128, 8, 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            # 256, 8, 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),

            # 256, 4, 4
            nn.Flatten(),

            # 256 * 4 * 4 = 4096
            nn.Linear(4096, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.Softmax(dim=1),
            nn.Linear(64, 8))
    
    def forward(self, xb):
        return self.model(xb)










dataset = ImageFolder('./dataset/', transform=T.ToTensor())
print(type(dataset))
dataset = [(T.functional.resize(a,128),b) for a, b in dataset]
print(type(dataset))
train_ds, val_ds, test_ds = random_split(dataset,[0.75,0.15,0.1])
print((len(train_ds) , len(val_ds) , len(test_ds)))
print()

batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size)
model = Classifier()

for images, labels in train_dl:
    out = model(images)
    print(out[0])#, out)
    print(labels)
    break







