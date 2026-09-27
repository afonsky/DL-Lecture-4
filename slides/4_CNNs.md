---
layout: center
---

<center>

# LeNet

# Putting it all together
</center>

---
zoom: 0.84
---

# LeNet-5 [LeCun et al., 1998]

<div class="grid grid-cols-[5fr_7fr] gap-6">
<div>

#### Historical significance:
* **1989**: LeCun et al. (Bell Labs) train a CNN with backprop on **US Postal Service zip codes**
* **LeNet-5** read bank checks in NCR machines from **1996** — about **10 % of all US checks** by 2001
* Precursor: Fukushima's **Neocognitron** (1980), no backprop

#### Design choices of the time:
* Sigmoid-type activations, **average** pooling; ReLU and max-pooling came later

</div>
<div>
<center>
  <figure>
    <img src="/lenet.svg" style="width: 540px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
      <a href="https://d2l.ai/chapter_convolutional-neural-networks/lenet.html">d2l.ai Fig. 7.6.1 Data flow in LeNet. The input is a handwritten digit, the output is a probability over 10 possible outcomes</a>
    </figcaption>
  </figure>   
</center>

#### Architecture:
* $2$ conv layers, $5\times5$, with $6$ and $16$ channels; $2$ pooling layers, $2\times2$, stride $2$
* $3$ fully-connected layers: $120, 84, 10$ — about $60\,000$ parameters
* **conv → pool → conv → pool → fully connected** is still the template

</div>
</div>

<span class="refs">Papers: [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) · [LeCun et al. (1989)](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) · [Fukushima (1980)](https://doi.org/10.1007/BF00344251) · Watch: [Convolutional Network Demo from 1989](https://www.youtube.com/watch?v=FwFduRA_L6Q) · Read: [d2l.ai 7.6](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)</span>

<!--
**Presenter Notes:**

**Historical importance of LeNet:**

- 1989 (LeNet-1 family): the first CNN trained end-to-end with backpropagation, on zip codes from the US Postal Service (Bell Labs).
- LeNet-5 went into NCR's check-reading machines in June 1996; by 2001 they read an estimated 20 million checks a day, about 10% of all US checks.
- d2l: some ATMs still run code that LeCun and Bottou wrote in the 1990s.
- Show 30 seconds of the 1989 video if time allows: real-time digit recognition on 1990s hardware.

**Architecture (simple by today's standards):**
- 2 conv layers (5×5 filters), 2 subsampling layers (2×2, stride 2), 3 fully-connected layers
- ~60,000 parameters (tiny!)
- The original subsampling layers had a trainable coefficient and bias per map, and C3 connected only some of the S2 maps; d2l (and our code) simplify both.

**Key insight:** The architecture pattern (conv-pool-conv-pool-fc) became the template for all future CNNs.
-->

---
zoom: 0.89
---

# LeNet in PyTorch

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

```python {all|2-5|6|7-9|11-14|all}
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),                         # 16 x 5 x 5 -> 400
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10))                    # logits, no softmax

X = torch.randn(1, 1, 28, 28)             # (batch, channels, height, width)
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__:<10} {tuple(X.shape)}")
```

</div>
<div class="compact-table">

| layer | printed shape | params |
|---|---|---|
| Conv2d | (1, 6, 28, 28) | 156 |
| MaxPool2d | (1, 6, 14, 14) | 0 |
| Conv2d | (1, 16, 10, 10) | 2,416 |
| MaxPool2d | (1, 16, 5, 5) | 0 |
| Flatten | (1, 400) | 0 |
| Linear | (1, 120) | 48,120 |
| Linear | (1, 84) | 10,164 |
| Linear | (1, 10) | 850 |
| **total** | | **61,706** |

<small>ReLU rows omitted: same shape, no parameters</small>

</div>
</div>

<v-clicks>

* **Channels up, resolution down**: $1 \to 6 \to 16$ channels while $28 \to 14 \to 10 \to 5$
* 78 % of the parameters are in the first fully connected layer; the training loop is **Lecture 3's**, unchanged

</v-clicks>

<style>
.compact-table table { font-size: 0.78em; }
.compact-table td, .compact-table th { padding-top: 0.2em; padding-bottom: 0.2em; }
</style>

<span class="refs">Read: [d2l.ai 7.6.1](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html#lenet) · Watch: [S. Raschka, L13.9.1 LeNet-5 in PyTorch](https://www.youtube.com/watch?v=ye5k82FQC7I) · Docs: [nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [nn.MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)</span>

<!--
d2l's LeNet with the two modern swaps from its exercise 1: ReLU instead of sigmoid, max- instead of average pooling.
The shapes are what this loop prints (PyTorch 2.14); check two of them with the formula:
conv1 (28 + 4 - 5)/1 + 1 = 28, conv2 (14 - 5)/1 + 1 = 10. Parameters: conv1 (25·1 + 1)·6 = 156, conv2 (25·6 + 1)·16 = 2,416.
Flatten turns the 16x5x5 block into a 400-vector - the only line an MLP person has to think about.
Point back to Lecture 3's slide "An MLP in PyTorch": opt, loss_fn and the loop are identical; only `model` changed.
-->

---
zoom: 0.96
---

# LeNet vs. Lecture 3's MLP on Fashion-MNIST

<figure>
  <img src="/lenet_vs_mlp.svg" style="width: 900px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Same data, loop and optimizer (AdamW, lr 10⁻³, weight decay 10⁻², batch 128, 10 epochs, no augmentation); mean of 3 seeds, bands show min–max.
  </figcaption>
</figure>

<div class="grid grid-cols-3 gap-6" style="font-size: 0.8em">
<div>

**(a)** Better with **3.3× fewer parameters** — but **2× more** multiply-adds per image (0.42 M vs. 0.20 M)

</div>
<div>
<v-click>

**(b)** More robust to **small** shifts: at 2 px, 81 % vs. 67 %. Both collapse by 4–6 px: the fully connected head still sees *where*

</v-click>
</div>
<div>
<v-click>

**(c&#41;** Shuffle the pixels (one fixed permutation): the MLP does not notice, LeNet loses 4 points — **its prior no longer holds**

</v-click>
</div>
</div>

<span class="refs">Code: [d2l.ai 7.6.2 Training](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html#training) · Try it in the browser: [A. Karpathy, ConvNetJS MNIST demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)</span>

<!--
Measured for this lecture (PyTorch 2.14, CPU, ~40 s per LeNet run). Per seed, final test accuracy:
MLP 88.56 / 88.35 / 88.53, LeNet 90.26 / 89.77 / 89.85.
Shifted test images, mean over the four directions, 0..6 px:
MLP 88.5, 82.6, 66.9, 51.4, 39.8, 33.4, 27.8; LeNet 90.0, 87.4, 80.7, 67.8, 51.4, 41.1, 34.0.
Shuffled pixels: MLP 88.6 (unchanged), LeNet 85.9 (-4.1).
Multiply-adds per image: LeNet 117,600 + 240,000 + 48,000 + 10,080 + 840 = 416,520; MLP 200,704 + 2,560 = 203,264.

If asked why LeNet also collapses at 4+ px (one seed each, same budget):
- replace the fully connected head by global average pooling (Lecture 5): 88.2% clean, 77% at 4 px (LeNet: 51%);
- or train LeNet on randomly shifted images (+-3 px, data augmentation, Lecture 5): 87.8% clean, 83% at 4 px.
Invariance comes from the architecture AND from the data.

Panel (c) is the single best argument for "inductive bias": the prior helps exactly when it is true of the data.
It is also Abu-Mostafa's point about constraints in the direction of the target function (Lecture 12).
-->

---

# Convolutional Neural Network (CNN)
### CNN is a sequence of convolutional layers, interspersed with activation functions
<br>
<br>
<br>
<div>
  <figure><center>
    <img src="/cnn_layers.png" style="width: 700px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image by
      <a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf">Andrej Karpathy</a>
    </figcaption>
  </figure>   
</div>

<!--
**Presenter Notes:**

**CNN = stacking convolutional layers**

Key points from Karpathy's visualization:
- A 32x32x3 input, 6 filters of 5x5x3 -> 28x28x6, then 10 filters of 5x5x6 -> 24x24x10 (no padding: each layer loses 4 pixels)
- Each layer extracts different features
- Spatial dimensions typically decrease
- Number of channels typically increases
- Each filter's depth equals the number of input channels (3, then 6)

**The pattern:**
Conv → ReLU → Conv → ReLU → Pool → ... → FC → Softmax
-->

---
zoom: 0.88
---

# The Standard Recipe

<div class="grid grid-cols-[3fr_2fr] gap-8">
<div>

### The most common pattern *(CS231n)*

```text
INPUT → [[CONV → RELU]*N → POOL?]*M → [FC → RELU]*K → FC
```
<small>usually $N \le 3$ and $K < 3$. LeNet has $N = 1$, $M = 2$, $K = 2$</small>

### Rules of thumb
* **CONV**: small kernels ($3\times3$, at most $5\times5$), stride 1, "same" padding — the size stays put
* **POOL**: $2\times2$, stride 2 — where the size shrinks
* **Channels up, resolution down**: e.g. $64 \to 128 \to 256$ while $224 \to 112 \to 56$
* Two stacked $3\times3$ layers see $5\times5$, with fewer weights<br> ($18c^2$ vs. $25c^2$) and one more ReLU

</div>
<div>

<v-click>

### Copy before you invent *(Lecture 3)*
* Start from a published architecture and change one thing at a time
* Lecture 5 is the catalogue: AlexNet, VGG, NiN, GoogLeNet, ResNet, DenseNet

</v-click>

</div>
</div>

<span class="refs">Read: [CS231n, Layer patterns](https://cs231n.github.io/convolutional-networks/#layer-patterns) · [Layer sizing patterns](https://cs231n.github.io/convolutional-networks/#layer-sizing-patterns) · Watch: [A. Ng, C4W1L10 CNN Example](https://www.youtube.com/watch?v=bXJx7y51cl0)</span>

<!--
The pattern is quoted from Karpathy's CS231n notes. It covers everything from a linear classifier (N = M = K = 0)
to VGG. Why stack small kernels: three 3x3 layers see 7x7 with 27c^2 weights instead of 49c^2, and three nonlinearities
instead of one - CS231n: "Prefer a stack of small filter CONV to one large receptive field CONV layer".
Keep the doubling-channels/halving-size rhythm in mind: the compute per layer stays roughly constant.
-->

---

# 2D convolutional NN visualization on MNIST <a href="https://adamharley.com/nn_vis/cnn/2d.html">[link]</a>

<iframe src="https://adamharley.com/nn_vis/cnn/2d.html" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>

<!--
**Presenter Notes:**

**Interactive demo** - spend 2-3 minutes here:

1. Draw different digits and watch activations change
2. Point out:
   - First conv layer: edge-like patterns
   - Deeper layers: more abstract features
   - Final layer: class probabilities

**Ask students:**
- What happens when you draw a partial digit?
- Which neurons activate for "1" vs "7"?
- Can you see the spatial reduction through pooling?
- Draw a digit in a corner: does it still work? (Link back to the shift experiment.)

**If internet is slow:** use the 3D version (https://adamharley.com/nn_vis/cnn/3d.html) or skip; the LeNet slides carry the content.
-->

---
layout: center
---

<center>

# What Do CNNs Learn?

# Receptive fields and feature hierarchies
</center>

---
zoom: 0.9
---

# Receptive Field

<figure>
  <img src="/receptive_field.svg" style="width: 880px !important; margin: 0 auto;">
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Receptive field**: the input region that can affect a unit
* Each $k\times k$ layer adds $k-1$ pixels **times the stride so far** — pooling multiplies the growth
* LeNet: $5 \to 6 \to 14 \to 16$ — each unit of the last map sees $16\times16$ of the $28\times28$ digit

</div>
<div>

<v-click>

* Term from neuroscience: **Hubel & Wiesel** found cells in the cat's visual cortex that respond to edges in small regions ([1962](https://doi.org/10.1113/jphysiol.1962.sp006837); Nobel Prize 1981)
* The **effective** receptive field is smaller: central pixels dominate ([Luo et al., 2016](https://arxiv.org/abs/1701.04128))

</v-click>
</div>
</div>

<span class="refs">Read: [d2l.ai 7.2.6 Feature Map and Receptive Field](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#feature-map-and-receptive-field) · [CS231n, "Prefer a stack of small filter CONV"](https://cs231n.github.io/convolutional-networks/#layer-patterns)</span>

<!--
**Presenter Notes:**

**Receptive field is crucial to understand:**

**Definition:** The region in the INPUT that affects a particular output neuron.

**How it grows (1-D view, same in 2-D):**
- (a) three 3-wide convolutions: 3, 5, 7 - each layer adds k - 1 = 2
- (b) conv, 2x2 pool with stride 2, conv: 3, 4, 8 - after the stride-2 pool, the next 3-wide kernel adds (3-1) x 2 = 4
- Rule: r_new = r_old + (k - 1) x (product of all earlier strides)
- LeNet: conv5 -> 5, pool2 -> 6, conv5 -> 6 + 4·2 = 14, pool2 -> 16

**Why it matters:**
- To detect a face, network must "see" entire face
- If receptive field is too small, can only see edges
- This is why we need DEEP networks (or downsampling)

**Biological connection:**
- Hubel & Wiesel recorded single neurons in the cat's visual cortex (1959, 1962)
- Nobel Prize in Physiology or Medicine 1981
- Neurons respond to oriented edges in specific visual field regions
-->

---
zoom: 0.9
---

# What Do CNNs Learn?

<figure>
  <img src="/lenet_features.svg" style="width: 880px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Our LeNet from the previous slides (seed 0, 10 epochs on Fashion-MNIST) on one test image. Kernels: blue positive, red negative weights.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Layer 1** of our small LeNet: two maps light up on the background around the silhouette, the others on edges of one orientation — sole, heel, top line
* **Layer 2** combines them: $16$ coarser maps, $10\times10$, each seeing $14\times14$ pixels

</div>
<div>

<v-click>

* Large CNNs on ImageNet: layer 1 = **Gabor-like** edges and colour blobs, as in the visual cortex (V1); deeper = textures → parts → objects ([Zeiler & Fergus, 2014](https://arxiv.org/abs/1311.2901))
* Explore: [CNN Explainer](https://poloclub.github.io/cnn-explainer/) · [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/)

</v-click>
</div>
</div>

<span class="refs">Watch: [S. Raschka, L13.8 What a CNN Can See](https://www.youtube.com/watch?v=PRFP5YC3u7g) · [Y. LeCun, NYU Deep Learning, Week 3](https://atcold.github.io/NYU-DLSP20/en/week03/03-1/)</span>

<!--
**Presenter Notes:**

Honest reading of our own model: with only 6 kernels on 28x28 grayscale clothing, layer 1 does not produce textbook
Gabor filters. Two kernels respond to the (dark) background, which outlines the silhouette; the rest respond to edges of
particular orientations. Layer 2 maps are coarser and combine them.

**The remarkable thing (large networks):**
- We didn't design these features - the network LEARNED them from data
- On ImageNet, first-layer filters look like Gabor filters and colour blobs, strikingly similar to V1 simple cells
- Deeper layers respond to textures, then object parts, then whole objects (Zeiler & Fergus used deconvolution to show this)
- And these features are useful for many tasks (transfer learning, Lecture 5 and later)

**Demo:** CNN Explainer if time permits - it runs a small CNN in the browser and shows every intermediate map.
-->

---

# CNN for Deep Learning
## Deep Learning = Learning Hierarchical Representations
### It's deep if it has more than one stage of non-linear feature transformation
<br>
<div class="grid grid-cols-[3fr_2fr] gap-4">
<div>
  <figure><center>
    <img src="/cnn_hierarchical_representation.png" style="width: 450px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image by
      <a href="https://drive.google.com/file/d/18UFaOGNKKKO5TYnSxr2b8dryI-PgZQmC/view?usp=share_link">Yann LeCun</a>
    </figcaption>
  </figure>   
</div>
<div>

### Feature Hierarchy:
1. **Layer 1**: Edges, colors, gradients
2. **Layer 2**: Textures, patterns
3. **Layer 3**: Parts (eyes, wheels)
4. **Layer 4+**: Objects, scenes

> *"Deep neural networks exploit the property that many natural signals are compositional hierarchies, in which higher-level features are obtained by composing lower-level ones."*
> <small>— LeCun, Bengio & Hinton, [Deep learning](https://www.nature.com/articles/nature14539), *Nature* (2015)</small>

</div>
</div>

<!--
**Presenter Notes:**

**Deep Learning = Hierarchical Feature Learning**

**LeCun's insight:**
- Each layer builds on the previous
- Simple features combine into complex ones
- Like building with LEGO bricks (the first slide of today)

**The hierarchy (from LeCun's visualization, features from Zeiler & Fergus):**
1. Edges and colors
2. Textures and simple shapes
3. Object parts (eyes, wheels, windows)
4. Whole objects (faces, cars)

**Why "deep" matters:**
- More layers = more abstraction, and a larger receptive field
- Each layer adds non-linearity
- Can represent increasingly complex functions

Same paper: "higher layers of representation amplify aspects of the input that are important for discrimination and
suppress irrelevant variations" - lighting, small shifts, background.
-->

---

# Convolutional Neural Network
## Putting it all together
<br>
<div>
  <figure><center>
    <img src="/cnn_architecture.jpg" style="width: 700px !important;">
</center>
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image by
      <a href="http://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf">Andrej Karpathy</a>
    </figcaption>
  </figure>   
</div>

<!--
Read it left to right with the class: CONV-RELU-CONV-RELU-POOL three times, then FC; the activation maps get smaller
and more abstract, and the last layer's scores are the logits for car, truck, airplane, ship, horse.
This is the CS231n pattern from "The Standard Recipe" with N = 2, M = 3, K = 0.
-->
