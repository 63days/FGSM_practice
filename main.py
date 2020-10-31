import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from model import Net
import torch.optim as optim
from tqdm import tqdm
from dataloader import DataSetWrapper
import torchvision.models as models
wrapper = DataSetWrapper(32, 4, 0.2)

train_dl = wrapper.get_train_data_loader()
test_dl = wrapper.get_test_data_loader()

# model = Net()
model = models.resnet50(pretrained=True, progress=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 10)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses=[]
val_losses=[]

for epoch in range(100):
    train_loss = []
    pbar = tqdm(train_dl)
    for img, label in pbar:
        optimizer.zero_grad()
        batch_size = img.size(0)
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        acc = (pred.argmax(1) == label).float().sum().item() / batch_size

        pbar.set_description(f'E: {epoch+1:3} L: {loss.item():.4f} A: {100*acc:.1f}')
        train_loss.append(loss.item())
    train_loss = sum(train_loss) / len(train_loss)
    train_losses.append(train_loss)


torch.save({
    'model_state_dict': model.state_dict(),
    'train_losses': train_losses
}, 'resnet50.pt')

pbar = tqdm(test_dl)
accs = []
with torch.no_grad():
    for img, label in test_dl:
        batch_size = img.size(0)
        pred = model(img)
        pred = pred.argmax(1)

        acc = (pred == label).float().sum().item() / batch_size

        accs.append(acc)

    acc_mean = sum(accs) / len(accs)

    print(f'test accuracy: {100 * acc_mean:.1f}%')






