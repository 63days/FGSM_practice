import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import Net
import torch.optim as optim
from tqdm import tqdm
from dataloader import DataSetWrapper

wrapper = DataSetWrapper(32, 8, 0.2)
test_dl = wrapper.get_test_data_loader()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model = Net()
load_state = torch.load('checkpoint.pt')
model.load_state_dict(load_state['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()

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


fgsm_accs = []

for img, label in test_dl:
    batch_size = img.size(0)
    img.requires_grad = True
    pred = model(img)
    softmax_pred = F.softmax(pred, dim=1)
    ori_prob = softmax_pred.max(1)[0]
    loss = criterion(pred, label)
    loss.backward()

    epsilon = 1e-1
    n = epsilon * torch.sign(img.grad)

    img = img + n
    pred = model(img)
    softmax_pred = F.softmax(pred, dim=1)
    pred = pred.argmax(1)
    prob = softmax_pred.max(1)[0]
    print(f'{classes[label[0]]} {100*ori_prob[0].item():.1f}% | {classes[pred[0]]} {100*prob[0].item():.1f}%')
    acc = (pred == label).float().sum().item() / batch_size
    fgsm_accs.append(acc)

fgsm_acc_mean = sum(fgsm_accs) / len(fgsm_accs)

print(f'FGSM test accuracy: {100 * fgsm_acc_mean:.1f}%')





