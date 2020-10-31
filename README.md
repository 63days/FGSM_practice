# Fast Gradient Sign Method: Pytorch Implementation
This repo is FGSM(Fast Gradient Sign Method) example implemented by pytorch.

## What is the FGSM?
Modern machine learning models are vulnerable to adversarial examples. Adversarial example is made deliberately to cause the model to malfunction by adding imperceptible perturbation to the origin data. In the image below, origin data(Panda) is slightly changed as on the right by noise. In the human eyes right image still looks like panda, but our machine learning model sees as gibbon. Like this, FGSM is the method generating adversarial examples easily and quickly.
![image](https://user-images.githubusercontent.com/37788686/97774252-89c0eb00-1b99-11eb-9af1-f213c0b89a9d.png)

## FGSM Algorithm
![image](https://user-images.githubusercontent.com/37788686/97774441-246df980-1b9b-11eb-9e00-761bc22ad6f0.png)

 
## Some Results
![image](https://user-images.githubusercontent.com/37788686/97774469-5d0dd300-1b9b-11eb-8c90-5046530c1994.png)
| model | dataset | epsilon |
| ---- | - | - |
| Resnet50 | CIFAR10 | 0.01 |

## Reference
[Explaining and Harnessing Adversarial Examples(ICLR'15), Ian Goodfellow](https://arxiv.org/abs/1412.6572)
