# Fast Gradient Sign Method: Pytorch Implementation
This repo is FGSM(Fast Gradient Sign Method) example implemented by pytorch.

## What is the FGSM?
Modern machine learning models are vulnerable to adversarial examples. Adversarial example is made deliberately to cause the model to malfunction by adding imperceptible perturbation to the origin data. In the image below, origin data(Panda) is slightly changed as on the right by noise. In the human eyes right image still looks like panda, but our machine learning model sees as gibbon. Like this, FGSM is the method generating adversarial examples easily and quickly.
![image](https://user-images.githubusercontent.com/37788686/97774252-89c0eb00-1b99-11eb-9af1-f213c0b89a9d.png)

## FGSM Algorithm
<img src="https://user-images.githubusercontent.com/37788686/97774441-246df980-1b9b-11eb-9e00-761bc22ad6f0.png" width="60%">

eta is a perturbation invisible to human but breaks our model. eta is calculated by gradients w.r.t input and sign function.

epsilon is hyperparameter of how much strong perturbation is. If epsilon is too big, perturbation is visible to human eye.
 
## Some Results
<img src="https://user-images.githubusercontent.com/37788686/97774469-5d0dd300-1b9b-11eb-8c90-5046530c1994.png" width="90%">

| model | dataset | epsilon |
| ---- | - | - |
| Resnet50 | CIFAR10 | 0.01 |

## Code Explanation
* main.py
```python
model = models.resnet50(pretrained=True, progress=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 10)
```
I used pretrained resnet50 as baseline. To classify CIFAR10 dataset, I changed resnet's fc to `nn.Linear(2048, 10)`. 

* test.py
```python
for img, label in test_dl:
    batch_size = img.size(0)
    img.requires_grad = True
    pred = model(img)
    softmax_pred = F.softmax(pred, dim=1)
    ori_prob = softmax_pred.max(1)[0]
    loss = criterion(pred, label)
    loss.backward()

    epsilon = 1e-1
    n = epsilon * torch.sign(img.grad) # Perturbation

    img = img + n
    pred = model(img)
    softmax_pred = F.softmax(pred, dim=1)
    pred = pred.argmax(1)
    prob = softmax_pred.max(1)[0]
    #print(f'{classes[label[0]]} {100*ori_prob[0].item():.1f}% | {classes[pred[0]]} {100*prob[0].item():.1f}%')
    acc = (pred == label).float().sum().item() / batch_size
    fgsm_accs.append(acc)

fgsm_acc_mean = sum(fgsm_accs) / len(fgsm_accs)

print(f'FGSM test accuracy: {100 * fgsm_acc_mean:.1f}%')
```
To get gradients w.r.t image, I set `img.requires_grad = True`. After `loss.backward()`, gradients of image is calculated and save in `img.grad`. `n = epsilon * torch.sign(img.grad)` is a perturbation breaking our model. `epsilon=1e-1` is smaller than precise of pixel. 

`img=img+n` <- This makes an adversarial example. 
## Reference
[Explaining and Harnessing Adversarial Examples(ICLR'15), Ian Goodfellow](https://arxiv.org/abs/1412.6572)




For more details, https://www.youtube.com/watch?v=YgmhBPLWo8I&t=497s
