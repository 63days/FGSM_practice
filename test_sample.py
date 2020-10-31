import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import Net
import torch.optim as optim
from tqdm import tqdm
from dataloader import DataSetWrapper
import torchvision.models as models
import matplotlib.pyplot as plt

transform = transforms.Compose(
            [transforms.ToTensor(),
             #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = models.resnet50()
model.fc = nn.Linear(2048, 10)

load_state = torch.load('resnet50.pt', map_location='cpu')
model.load_state_dict(load_state['model_state_dict'])

for param in model.parameters():
    param.requires_grad = False
model.eval()
criterion = nn.CrossEntropyLoss()
epsilon = 1e-2

sample = (iter(testset))

next(sample)
next(sample)
next(sample)
sample = next(sample)
img, label = sample
img = img.unsqueeze(0)
img.requires_grad = True
label = torch.tensor([label])

pred = model(img)

loss = criterion(pred, label)
loss.backward()
n = epsilon * torch.sign(img.grad)
perturbed_img = img + n

perturbed_pred = model(perturbed_img)

pred = F.softmax(pred, dim=1)
pred_label, pred_prob = pred.argmax(1).item(), pred.max(1)[0].item()

perturbed_pred = F.softmax(perturbed_pred, 1)
perturbed_label, perturbed_prob = perturbed_pred.argmax(1).item(), perturbed_pred.max(1)[0].item()

print(f'{classes[pred_label]} {pred_prob:.2f} | {classes[perturbed_label]} {perturbed_prob:.2f}')

fig, ax = plt.subplots(1,3)

ax[0].imshow(transforms.ToPILImage()(img.squeeze(0)))
ax[0].axis('off')
ax[0].set_title('Original')
ax[0].text(20,34,f'{classes[pred_label]} {100*pred_prob:.1f}%')
ax[1].imshow(transforms.ToPILImage()(n.squeeze(0)))
ax[1].axis('off')
ax[1].set_title('Perturbation')
ax[2].imshow(transforms.ToPILImage()(perturbed_img.squeeze(0)))
ax[2].axis('off')
ax[2].set_title('Adversarial Example')
ax[2].text(20,34,f'{classes[perturbed_label]} {100*perturbed_prob:.1f}%')
plt.show()

