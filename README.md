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

    

  *  <div align=center>
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

    

  -  <div align=center>
         <img src="https://latex.codecogs.com/gif.latex?maximize_{\delta&space;\in&space;\Delta}&space;(\ell(h_\theta(x&space;&plus;\delta),&space;y)-&space;\ell(h_\theta(x&space;&plus;\delta),&space;y_{\mathrm{target}})&space;\equiv&space;maximize_{\delta&space;\in&space;\Delta}(h_\theta(x&plus;\delta)_{y_{\mathrm{target}}}-&space;h_\theta(x&plus;\delta)_{y}&space;)" />
    </div> 




## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
