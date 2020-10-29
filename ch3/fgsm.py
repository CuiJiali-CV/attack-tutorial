import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import model_dnn_2, model_cnn, model_dnn_4

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for X, y in test_loader:
    X, y = X.to(device), y.to(device)
    break

def plot_images(X, y, yp, M, N, dir):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M * 1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1 - X[i * N + j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i * N + j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i * N + j].max(dim=0)[1] == y[i * N + j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()
    plt.savefig(dir)

model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))
model_dnn_4.load_state_dict(torch.load("model_dnn_4.pt"))
model_cnn.load_state_dict(torch.load("model_cnn.pt"))

yp = model_dnn_2(X)
plot_images(X, y, yp, 3, 6, './dnn2_nat.png')

delta = fgsm(model_dnn_2, X, y, 0.1)
yp = model_dnn_2(X + delta)
plot_images(X+delta, y, yp, 3, 6, './dnn2_attack.png')

delta = fgsm(model_cnn, X, y, 0.1)
yp = model_cnn(X + delta)
plot_images(X+delta, y, yp, 3, 6, './cnn_attack.png')