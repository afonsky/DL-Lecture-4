---
layout: center
---

# Backup Slides

---
zoom: 0.86
---

# AlexNet [Krizhevsky et al., 2012]

<div>
  <figure><center>
    <img src="/cnn_AlexNet.png" style="width: 600px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;">Krizhevsky, Sutskever, and Hinton. "ImageNet classification with deep convolutional neural networks." NeurIPS 2012
    </figcaption>
  </figure>   
</div>
<br>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### The "ImageNet moment"
* Won ImageNet 2012 with **15.3% error**<br> (vs 26.2% runner-up)
* Sparked the **deep learning revolution**
* ~60 million parameters

</div>
<div>

### Key innovations:
* **ReLU activation** (faster training than tanh)
* **Dropout** for regularization
* **GPU training** (2× NVIDIA GTX 580)
* **Data augmentation** (crops, flips)
* **Local Response Normalization** (rarely used now)

</div>
</div>

<span style="color:grey"><small>Often credited as the paper that launched the current era of deep learning</small></span>

---
zoom: 0.8
---

# VGG-16 [Simonyan et al., 2014]

<div class="grid grid-cols-[3fr_2fr] gap-4">
<div>
  <figure><center>
    <img src="/cnn_VGG-16.png" style="width: 550px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Simonyan & Zisserman. "Very deep convolutional networks for large-scale image recognition." (2014)
    </figcaption>
  </figure>   
</div>
<div>

### Key design principle:
**Use only $3\times 3$ convolutions!**

### Why $3 \times 3$?
* Two $3\times 3$ layers have same receptive field as one $5\times 5$
* But: fewer parameters & more non-linearity
* Three $3\times 3$: same as $7\times 7$ but with $3\times$ non-linearities

<br>

### VGG-16 specs:
* 16 weight layers (13 conv + 3 FC)
* **138 million** parameters
* Very regular structure
* 7.3% top-5 error on ImageNet

</div>
</div>

<span style="color:grey"><small>VGG networks are still widely used as **feature extractors** in transfer learning</small></span>

---
zoom: 0.9
---

# ResNet [He et al., 2015]

<div class="grid grid-cols-[2fr_3fr] gap-15">
<div>
  <figure><center>
    <img src="/cnn_ResNet.jpg" style="width: 350px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">He et al. "Deep residual learning for image recognition." CVPR (2016)
    </figcaption>
  </figure>   

<br>

> *"Skip connections are one of the most important innovations in deep learning."* — Sebastian Raschka
</div>
<div>

### The problem: Degradation
* Deeper networks should be better, right?
* In practice: very deep nets performed **worse**!
* Not overfitting — even training error was higher

<br>

### The solution: Skip Connections
* Learn $F(x) = H(x) - x$ (the **residual**)
* Output: $H(x) = F(x) + x$
* If identity is optimal, just learn $F(x) = 0$

<br>

### Impact:
* Enabled training of **152+ layer** networks
* Won ImageNet 2015: **3.57%** top-5 error
* Foundation for most modern architectures

</div>
</div>

---
zoom: 0.9
---

# CNN Evolution: Key Lessons

<div class="grid grid-cols-[1fr_1fr] gap-6">
<div>

### Design patterns that emerged:

| Architecture | Key Innovation |
|-------------|----------------|
| LeNet | Convolutions work! |
| AlexNet | ReLU, Dropout, GPUs |
| VGG | Deeper + small filters |
| ResNet | Skip connections |

### Trends over time:
* **Deeper** networks (5 → 152+ layers)
* **Smaller** filters ($11\times 11$ → $3\times 3$)
* **More regularization** (dropout, batch norm)
* **Better training** (better optimizers, data aug)

</div>
<div>

### Andrew Ng's practical advice:

1. **Start simple**: Use proven architectures
2. **Transfer learning**: Don't train from scratch
3. **Data augmentation**: Free performance boost
4. **Regularization**: BatchNorm > Dropout for CNNs

<br>

### What's next? (separate lecture)
* Inception/GoogLeNet
* DenseNet
* EfficientNet
* Vision Transformers (ViT)

</div>
</div>

---

# State of the art

### Finding right architectures: Active area or research

<div>
  <figure><center>
    <img src="/cnn_sota_1.png" style="width: 550px !important;">
</center>
  </figure>   
</div>
<br>

### Modular building blocks engineering

#### See also: [DenseNets](https://d2l.ai/chapter_convolutional-modern/densenet.html), [ResNet flavours](https://d2l.ai/chapter_convolutional-modern/resnet.html) (Wide ResNets, ResNeXts, Pyramidal ResNets)

<span style="color:grey"><small>From Kaiming He slides "Deep residual learning for image recognition." ICML. (2016)</small></span>

---

# State of the art

### Top 1-accuracy, performance and size on ImageNet

<div>
  <figure><center>
    <img src="/cnn_sota_2.png" style="width: 750px !important;">
</center>
  </figure>   
</div>
<br>

#### See also: https://paperswithcode.com/sota/image-classification-on-imagenet

<span style="color:grey"><small>From Canziani, Paszke, and Culurciello. "An Analysis of Deep Neural Network Models for Practical Applications." (May 2016)</small></span>