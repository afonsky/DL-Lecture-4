---
zoom: 0.84
---

# Conclusions: The Convolutional Layer

<div class="grid grid-cols-2 gap-10">
<div>

### Why
* Images have **local motifs** and **location-independent statistics** → **locality** and **weight sharing**
* A conv layer = a fully connected layer with most weights **zero** and the rest **shared**: $10^{12} \to 9$ parameters

### How
* Slide a $k\times k$ kernel, multiply elementwise, sum, add the bias: a **cross-correlation**
* Kernels are **learned** by backprop; edge detectors emerge on their own
* **Equivariant**: shift the input, the feature maps shift too

</div>
<div>

### Shapes and counts
* $n_\text{out} = \left\lfloor (n + 2p - k)/s \right\rfloor + 1$; "same" padding $p = (k-1)/2$
* Weights $c_\text{out}\times c_\text{in}\times k\times k$: $(k^2 c_\text{in} + 1)\,c_\text{out}$ parameters, **whatever the image size**
* Compute **does** grow with the image: every weight is used at every position
* $1\times1$ convolution = a fully connected layer across channels, at every pixel

</div>
</div>

---
zoom: 0.88
---

# Conclusions: Building a CNN

<div class="grid grid-cols-2 gap-10">
<div>

### Pooling
* Max or average over a window, **per channel**, **no parameters**
* Approximate invariance to **small** shifts; a faster-growing receptive field

### LeNet and the recipe
* `[CONV → ReLU]*N → POOL`, repeated, then fully connected layers → logits
* Channels up, resolution down; the training loop is Lecture 3's

</div>
<div>

### Measured on Fashion-MNIST
* LeNet vs. MLP: **90.0 % vs. 88.5 %** with **3.3× fewer** parameters
* Much more robust to 1–3 px shifts, **worse** on shuffled pixels

### What they learn
* Edges → textures → parts → objects, as the receptive field grows

</div>
</div>

> *"There are four key ideas behind ConvNets that take advantage of the properties of natural signals: local connections, shared weights, pooling and the use of many layers."* <small>— LeCun, Bengio & Hinton, [Deep learning](https://www.nature.com/articles/nature14539), *Nature* (2015)</small>

#### The one idea to take away: **the architecture is the prior.** Convolutions help exactly as much as the data really is local and translation-invariant.

---
zoom: 0.64
---

# Learn More from the Experts

| Expert | Watch / read | Today's topics |
|---|---|---|
| Andrew Ng | [Deep Learning Specialization, Course 4, Week 1](https://www.coursera.org/learn/convolutional-neural-networks) · [YouTube playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF) (C4W1L01–L11) | edge detection, padding, stride, volumes, pooling |
| Yaser Abu-Mostafa | [Learning From Data](https://work.caltech.edu/lectures.html): [Lecture 7, The VC Dimension](https://www.youtube.com/watch?v=Dc0sr0kdBVI) · [Lecture 12, Regularization](https://www.youtube.com/watch?v=I-VfYXzC5ro) | why fewer, constrained parameters generalize |
| Yann LeCun | [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) · [NYU Deep Learning, Week 3](https://atcold.github.io/NYU-DLSP20/en/week03/03-1/) · [LeNet-5 demos](http://yann.lecun.com/exdb/lenet/) | convolutions from first principles, LeNet |
| Yoshua Bengio | [Deep Learning book, Ch. 9](https://www.deeplearningbook.org/contents/convnets.html) · [LeCun & Bengio (1995)](http://yann.lecun.com/exdb/publis/pdf/lecun-bengio-95a.pdf) | sharing, equivariance, pooling as a prior |
| Andrej Karpathy | [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/) · [CS231n 2016, Lecture 7](https://www.youtube.com/watch?v=LxfUGhug-iQ) · [ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) | output sizes, parameter counts, layer patterns |
| Sebastian Raschka | [STAT 453, L13: Introduction to CNNs](https://sebastianraschka.com/blog/2021/dl-course.html#l13-introduction-to-convolutional-neural-networks) · [ML with PyTorch and Scikit-Learn, Ch. 14 code](https://github.com/rasbt/machine-learning-book/tree/main/ch14) | CNNs and LeNet-5 in PyTorch |
| Josh Starmer | [StatQuest: Image Classification with Convolutional Neural Networks](https://www.youtube.com/watch?v=HGwBXDKFk9I) | slow, visual intuition |

#### Main text: [d2l.ai, Chapter 7 — Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html), sections 7.1–7.6

### Next lecture: modern CNNs — AlexNet, VGG, NiN, GoogLeNet, batch normalization, ResNet, DenseNet ([d2l.ai, Ch. 8](https://d2l.ai/chapter_convolutional-modern/index.html))
