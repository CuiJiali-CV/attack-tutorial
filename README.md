<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>






# Attack Tutorial



## Adversarial example

* #####  Pre-trained ResNet50 1000 classes.

  

  * <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch1/pig.jpg" height="224" width="224" >
    </div>

  * ```python
    pig_img = Image.open("pig.jpg")
    preprocess = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
    ])
    pig_tensor = preprocess(pig_img)[None,:,:,:]
    
    class Normalize(nn.Module):
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.mean = torch.Tensor(mean)
            self.std = torch.Tensor(std)
        def forward(self, x):
            return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
    
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = resnet50(pretrained=True)
    model.eval()
    
    # form predictions
    pred = model(norm(pig_tensor))
    
    import json
    with open("imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    print(imagenet_classes[pred.max(dim=1)[1].item()])
    ```

  * ```python
    hog
    0.9961252808570862
    ```

* ##### Create an untargeted adversarial example

  * ```python
    epsilon = 2. / 255
    
    delta = torch.zeros_like(pig_tensor, requires_grad=True)
    opt = optim.SGD([delta], lr=1e-1)
    
    for t in range(30):
        pred = model(norm(pig_tensor + delta))
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([341]))
        if t % 5 == 0:
            print(t, loss.item())
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
    
    print("True class probability:", nn.Softmax(dim=1)(pred)[0, 341].item())
    
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", imagenet_classes[max_class])
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())
    ```

  * ```python
    True class probability: 1.9019864794245223e-07
    Predicted class:  wombat
    Predicted probability: 0.9999523162841797
    ```

    

  * <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch1/wombat.jpg" height="224" width="224" >
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch1/wombat_delta.jpg" height="224" width="224" >
    </div>

    

  * <div align=center>
        <img src="https://latex.codecogs.com/gif.latex?maximize_{\delta&space;\in&space;\Delta}&space;\ell(h_\theta(x&space;&plus;\delta),&space;y)" />
    </div> 


     <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?\Delta&space;=&space;\{\delta&space;:&space;\|\delta\|_\infty&space;\leq&space;\epsilon\}" />
    </div> 

    

- **Create a targeted adversarial example**

  - ```python
    epsilon = 2. / 255
    delta = torch.zeros_like(pig_tensor, requires_grad=True)
    opt = optim.SGD([delta], lr=5e-3)
    
    for t in range(100):
        pred = model(norm(pig_tensor + delta))
        loss = (-nn.CrossEntropyLoss()(pred, torch.LongTensor([341])) +
                nn.CrossEntropyLoss()(pred, torch.LongTensor([404])))
        if t % 10 == 0:
            print(t, loss.item())
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        delta.data.clamp_(-epsilon, epsilon)
    
    import json
    with open("imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    
    max_class = pred.max(dim=1)[1].item()
    print("Predicted class: ", imagenet_classes[max_class])
    print("Predicted probability:", nn.Softmax(dim=1)(pred)[0,max_class].item())
    ```

  - ```python
    Predicted class:  airliner
    Predicted probability: 0.9759377837181091
    ```

  - <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch1/airline.jpg" height="224" width="224" >
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch1/airline_delta.jpg" height="224" width="224" >
    </div>

    

  - <div align=center>
        <img src="https://latex.codecogs.com/gif.latex?maximize_{\delta&space;\in&space;\Delta}&space;(\ell(h_\theta(x&space;&plus;\delta),&space;y)-&space;\ell(h_\theta(x&space;&plus;\delta),&space;y_{\mathrm{target}})&space;\equiv&space;maximize_{\delta&space;\in&space;\Delta}(h_\theta(x&plus;\delta)_{y_{\mathrm{target}}}-&space;h_\theta(x&plus;\delta)_{y}&space;)" />
    </div> 





## Robust Training (Two classes linear model)

* #####  Two Classes Linear model

  1. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?minimize_{W,b}&space;\frac{1}{|D|}\sum_{x,y&space;\in&space;D}&space;\max_{\|\delta\|&space;\leq&space;\epsilon}\ell(W(x&plus;\delta)&space;&plus;&space;b,&space;y)" />
     </div> 

     </br>

  2. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?maximize_{\|\delta\|&space;\leq&space;\epsilon}&space;\ell(w^T&space;(x&plus;\delta),&space;y)&space;\equiv&space;maximize_{\|\delta\|&space;\leq&space;\epsilon}&space;L(y&space;\cdot&space;(w^T(x&plus;\delta)&space;&plus;&space;b))." />
     </div> 

    </br>

  3. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?max_{\|\delta\|&space;\leq&space;\epsilon}&space;L&space;(y&space;\cdot&space;(w^T(x&plus;\delta)&space;&plus;&space;b)&space;)&space;=&space;L\left(&space;\min_{\|\delta\|&space;\leq&space;\epsilon}&space;y&space;\cdot&space;(w^T(x&plus;\delta)&space;&plus;&space;b)&space;\right)&space;=&space;L\left(y\cdot(w^Tx&space;&plus;&space;b)&space;&plus;&space;\min_{\|\delta\|&space;\leq&space;\epsilon}&space;y&space;\cdot&space;w^T\delta&space;\right)" />
     </div> 


     </br>

  4. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?\delta^\star&space;=&space;-&space;y&space;\epsilon&space;\cdot&space;\mathrm{sign}(w)" />
     </div> 


     </br>

  5. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?y&space;\cdot&space;w^T\delta^\star&space;=&space;y&space;\cdot&space;\sum_{i=1}&space;-y&space;\epsilon&space;\cdot&space;\mathrm{sign}(w_i)&space;w_i&space;=&space;-y^2&space;\epsilon&space;\sum_{i}&space;|w_i|&space;=&space;-\epsilon&space;\|w\|_1." />
     </div> 


     </br>

  6. <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?minimize_{w,b}&space;\frac{1}{D}\sum_{(x,y)&space;\in&space;D}&space;L&space;\left(y&space;\cdot&space;(w^Tx&space;&plus;&space;b)&space;-&space;\epsilon&space;\|w\|_*&space;\right&space;)" />
     </div> 

* **Train two classes linear model**

  * ```python
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    
    train_idx = mnist_train.targets <= 1
    mnist_train.data = mnist_train.data[train_idx]
    mnist_train.targets = mnist_train.targets[train_idx]
    
    test_idx = mnist_test.targets <= 1
    mnist_test.data = mnist_test.data[test_idx]
    mnist_test.targets = mnist_test.targets[test_idx]
    
    train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    
    # do a single pass over the data
    def epoch(loader, model, opt=None):
        total_loss, total_err = 0., 0.
        for X, y in loader:
            yp = model(X.view(X.shape[0], -1))[:, 0]
            loss = nn.BCEWithLogitsLoss()(yp, y.float())
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
    
            total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)
    
    model = nn.Linear(784, 1)
    opt = optim.SGD(model.parameters(), lr=1.)
    print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
    for i in range(10):
        train_err, train_loss = epoch(train_loader, model, opt)
        test_err, test_loss = epoch(test_loader, model)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    
    X_test = (test_loader.dataset.test_data.float()/255).view(len(test_loader.dataset),-1)
    y_test = test_loader.dataset.test_labels
    yp = model(X_test)[:,0]
    idx = (yp > 0) * (y_test == 0) + (yp < 0) * (y_test == 1)
    plt.imshow(1-X_test[idx][0].view(28,28).numpy(), cmap="gray")
    plt.title("True Label: {}".format(y_test[idx][0].data.item()))
    plt.savefig('./prediction.png')
    
    epsilon = 0.2
    delta = epsilon * model.weight.detach().sign().view(28,28)
    plt.imsave('./opt_delta.jpg',1-delta.numpy(), cmap="gray")
    ```

  * <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch2/prediction.png" height="300" width="400" >
    </div>

* **Create optimized delta** 

  * ```python
    epsilon = 0.2
    delta = epsilon * model.weight.detach().sign().view(28,28)
    plt.imsave('./opt_delta.jpg',1-delta.numpy(), cmap="gray")
    ```

  * <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch2/opt_delta.jpg" height="250" width="250" >
    </div>

* **Test with adversarial example**

  * ```python
    def epoch_adv(loader, model, delta):
        total_loss, total_err = 0.,0.
        for X,y in loader:
            yp = model((X-(2*y.float()[:,None,None,None]-1)*delta).view(X.shape[0], -1))[:,0]
            loss = nn.BCEWithLogitsLoss()(yp, y.float())
            total_err += ((yp > 0) * (y==0) + (yp < 0) * (y==1)).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset)
    
    print(epoch_adv(test_loader, model, delta[None,None,:,:]))
    ```

  * ```python
    Output
    err: 0.857210401891253
    ```

  * <div align="center">
        <img src="https://github.com/CuiJiali-CV/attack-tutorial/raw/main/ch2/opt_att_imgs.png" height="400" width="400" >
    </div>

* **Train two classes robust linear model**

  * ```python
    def epoch_robust(loader, model, epsilon, opt=None):
        total_loss, total_err = 0., 0.
        for X, y in loader:
            yp = model(X.view(X.shape[0], -1))[:, 0] - epsilon * (2 * y.float() - 1) * model.weight.norm(1)
            loss = nn.BCEWithLogitsLoss()(yp, y.float())
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
    
            total_err += ((yp > 0) * (y == 0) + (yp < 0) * (y == 1)).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)
    
    model = nn.Linear(784, 1)
    opt = optim.SGD(model.parameters(), lr=1e-1)
    epsilon = 0.2
    print("Rob. Train Err", "Rob. Train Loss", "Rob. Test Err", "Rob. Test Loss", sep="\t")
    for i in range(20):
        train_err, train_loss = epoch_robust(train_loader, model, epsilon, opt)
        test_err, test_loss = epoch_robust(test_loader, model, epsilon)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    ```

  * ```python
    Rob. Train Err	Rob. Train Loss		Rob. Test Err	Rob. Test Loss
    0.034189		0.135993			0.023641		0.098824
    ```

  

* **Test it with non-adversarial data set**

  * ```python
    from linear_model import epoch
    train_err, train_loss = epoch(train_loader, model)
    test_err, test_loss = epoch(test_loader, model)
    print("Train Err", "Train Loss", "Test Err", "Test Loss", sep="\t")
    print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")
    ```

  * ```python
    Train Err	Train Loss	Test Err	Test Loss
    0.006238	0.015268	0.003783	0.008271
    ```

    


## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
