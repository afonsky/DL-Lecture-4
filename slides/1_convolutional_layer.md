---
layout: center
---

<center>

# Convolutional Layer

# One small kernel, slid over the whole image
</center>

---
zoom: 0.95
---

# Convolutional Layer

### This scan-like approach is realized in **convolution layers** of NNs:

<div class="grid grid-cols-[4fr_3fr] gap-6">
<div>
<center>
  <figure>
    <img src="/conv_2D_1.gif" style="width: 450px !important;">
  </figure>
</center>

<br>

The **same** kernel is applied at every position — this is **weight sharing**. First output: $3\cdot0 + 3\cdot1 + 2\cdot2 + 0\cdot2 + 0\cdot2 + 1\cdot0 + 3\cdot0 + 1\cdot1 + 2\cdot2 = 12$

</div>
<div>

### Key terminology:
* <span style="color: #268BD2">**Input:**</span> $5\times 5$
* <span style="color: #1A6998">**Kernel / filter**</span>: $3\times 3$
  * Contains **learnable weights**
* <span style="color: #2AA098">**Output / feature map**</span>: $3\times 3$
  * Also called **activation map**
* Plus one learnable **bias**, added to every output

</div>
</div>

<span class="refs">GIF: [AMLD 2019 PyTorch workshop](https://github.com/theevann/amld-pytorch-workshop/blob/master/6-CNN.ipynb), animating Fig. 1.1 of [Dumoulin & Visin, A guide to convolution arithmetic](https://arxiv.org/abs/1603.07285) · Watch: [S. Raschka, L13.4 Convolutional Filters and Weight-Sharing](https://www.youtube.com/watch?v=ryJ6Bna-ZNU)</span>

<!--
**Presenter Notes:**

This is the **core operation** - make sure students understand each component:

**Terminology:**
- **Input**: The image or feature map from previous layer
- **Kernel/Filter**: Small matrix of learnable weights (typically 3×3 or 5×5)
- **Output/Feature Map**: Result of applying the kernel across the input

**The operation:**
1. Place the kernel at top-left corner
2. Element-wise multiply kernel with overlapping input region
3. Sum all products to get ONE output value (the worked sum on the slide: 12)
4. Slide the kernel and repeat

**Weight sharing insight:** The SAME kernel weights are used at EVERY position. This is what gives us translation equivariance and dramatically reduces parameters.
-->

---
zoom: 0.9
---

# Convolutional Layer (1D)
<div>
<center>
  <figure>
    <img src="/conv_1D_1.gif" style="width: 410px !important;">
  </figure>
</center>   
</div>

#### A convolution is an operation between two signals: input and kernel.

#### To get the convolution output of an input vector and a kernel:
- **Slide the kernel** at each different possible positions in the input
- For each position, perform the **element-wise product**<br> between the kernel and the corresponding part of the input
- **Sum** the result of the element-wise product

<span class="refs">GIF: [AMLD 2019 PyTorch workshop](https://github.com/theevann/amld-pytorch-workshop/blob/master/6-CNN.ipynb) · Watch: [3Blue1Brown, But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA)</span>

<!--
**Presenter Notes:**

Start with 1D to build intuition before moving to 2D:

**The three steps:**
1. **Slide**: Move the kernel across the input
2. **Multiply**: Element-wise multiplication at each position
3. **Sum**: Add up all products to get one output value

Check the GIF's first output with the class: 1·1 + 4·2 + (-1)·0 + 0·(-1) = 9. The output has W - w + 1 = 10 - 4 + 1 = 7 entries.

**Example on board:**
- Input: [1, 2, 3, 4, 5]
- Kernel: [1, 0, -1]
- First output: 1×1 + 2×0 + 3×(-1) = -2

This kernel computes a **discrete derivative** - it detects changes/edges!
-->

---

# Convolutions in 2D: Example

<br>
<br>
<br>
<center>
<figure>
  <img src="/convolutions_1.png" style="width: 500px !important;">
</figure>
</center>

---

# Convolutions in 2D: Example

<br>
<br>
<br>
<center>
<figure>
  <img src="/convolutions_2.png" style="width: 500px !important;">
</figure>
</center>

<!--
Let the class compute the remaining output cells: 9, 4 in the first row, 5, 7 in the second (7 is shown).
The output is 3×3 because a 2×2 window fits in 3 positions along each axis of a 4×4 input: 4 - 2 + 1 = 3.
-->

---
zoom: 0.99
---

# Cross-Correlation vs Convolution

<div class="grid grid-cols-[1fr_1fr] gap-8">
<div>

### Strictly speaking...
* What deep learning calls "convolution" is **cross-correlation**
* True convolution **flips the kernel** horizontally and vertically before sliding it

### Why it does not matter
* The kernel is **learned**: a layer doing true convolution simply learns the flipped kernel
* Same outputs, same accuracy. `nn.Conv2d` computes cross-correlation

</div>
<div>

**Cross-correlation** (what `nn.Conv2d` computes):
$$Y[i,j] = \sum_a \sum_b K[a,b] \cdot X[i+a, j+b]$$

**True convolution** (signal processing):
$$Y[i,j] = \sum_a \sum_b K[a,b] \cdot X[i-a, j-b]$$

> *"In keeping with standard terminology in deep learning literature, we will continue to refer to the cross-correlation operation as a convolution..."*
> <small>— [d2l.ai 7.2.5](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#cross-correlation-and-convolution)</small>

</div>
</div>

<span class="refs">Watch: [S. Raschka, L13.5 What's the difference between cross-correlation and convolution?](https://www.youtube.com/watch?v=xbO-iIzkBy0)</span>

<!--
**Presenter Notes:**

**Technical note** for the curious students:

- True mathematical convolution **flips** the kernel before sliding
- Deep learning "convolution" is actually **cross-correlation** (no flip)

**Why it doesn't matter:**
- The kernel is **learned from data**
- If we used true convolution, the network would just learn the flipped kernel
- Final result is identical

**Bottom line:** Don't worry about this distinction in practice. Everyone in DL calls it "convolution."
The flip matters only when you copy a hand-designed kernel from a signal-processing textbook.
-->

---
zoom: 0.95
---

# Kernels Are Pattern Detectors

<figure>
  <img src="/kernels_demo.png" style="width: 860px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    Each kernel cross-correlated with the same 160×160 photo (Grace Hopper, public domain). Edge maps: blue = positive, red = negative response.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Blur** = average of 9 neighbours; **sharpen** = boost the centre, subtract the neighbours
* **Edge** kernels sum to 0: silent on flat regions, loud where brightness **changes**

</div>
<div>

<v-click>

* Classical vision hand-designed such kernels for decades (Sobel, Canny, Gabor)
* **A CNN learns its kernels** — hundreds of them, tuned to the task

</v-click>
</div>
</div>

<span class="refs">Watch: [A. Ng, C4W1L02 Edge Detection Examples](https://www.youtube.com/watch?v=XuD4C8vJzEQ) · [C4W1L03 More Edge Detection](https://www.youtube.com/watch?v=am36dePheDc) · Read: [d2l.ai 7.2.3](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#object-edge-detection-in-images)</span>

<!--
Read the kernels with the class, left to right.
Identity: a single 1 in the centre copies the image - the kernel is just a set of weights.
Blur: every output is the mean of its 3x3 neighbourhood.
Sharpen: 5 x centre minus the 4 neighbours = the image plus its "details" (image minus its local average).
Vertical edges (Ng's example): left column minus right column; responds where brightness changes from left to right -
the flag stripes and the edges of the cap light up, flat regions are ~0. The sign tells the direction of the change.
Horizontal edges: the same kernel transposed.
The punchline is the second click: nobody designs these any more; they come out of gradient descent (next slides).
-->

---

# Image Kernels Explained Visually <a href="https://setosa.io/ev/image-kernels/">[link]</a>

<iframe src="https://setosa.io/ev/image-kernels/" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>

<!--
Live demo, 1-2 min. Scroll to the interactive part: pick "outline", then "sharpen", and hover over the
input to show the 3x3 window and the weighted sum - the same computation as on the previous slides.
Type your own kernel: [1 0 -1] rows give the vertical-edge detector.
-->

---
zoom: 0.96
---

# Learning a Kernel from Data

<div class="grid grid-cols-[3fr_2fr] gap-6">
<div>

```python {all|5-8|10|11-18|all}
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.ones(1, 1, 6, 8)            # (batch, channels, height, width)
X[..., 2:6] = 0                       # a black band on a white image
Y = F.conv2d(X, torch.tensor([[[[1.0, -1.0]]]]))   # targets: its edges

conv = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)   # random kernel
for i in range(10):
    loss = ((conv(X) - Y) ** 2).sum()
    conv.zero_grad()
    loss.backward()
    conv.weight.data -= 3e-2 * conv.weight.grad
    if (i + 1) % 2 == 0:
        print(f"epoch {i + 1}, loss {loss:.3f}")
print(conv.weight.data.reshape(1, 2))
```

</div>
<div>

```text
epoch 2, loss 8.330
epoch 4, loss 1.722
epoch 6, loss 0.422
epoch 8, loss 0.125
epoch 10, loss 0.043
tensor([[ 1.0063, -0.9662]])
```

<v-clicks>

* Only input–output pairs are given: gradient descent **finds** the edge detector $[1, -1]$
* **Backpropagation** *(Lecture 3)* through `nn.Conv2d` is automatic
* Real CNNs learn **thousands** of kernels this way

</v-clicks>
</div>
</div>

<span class="refs">Read: [d2l.ai 7.2.4 Learning a Kernel](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#learning-a-kernel) · Watch: [S. Raschka, L13.6 CNNs & Backpropagation](https://www.youtube.com/watch?v=-SwKNK9MIUU)</span>

<!--
d2l 7.2.4, executed (PyTorch 2.14, CPU): the output on the right is what this exact code prints.
Y has a +1 column where white turns black and a -1 column where black turns white; everything else is 0.
The loop is Lecture 2's training loop with a convolution as the model and squared error as the loss.
Point at the four lines forward - loss - backward - step: nothing about training changes for CNNs.
Backprop through a convolution is itself a convolution (with the flipped kernel) - frameworks do it for you.
-->

---
zoom: 0.96
---

# Convolutions: Key Properties

<div class="grid grid-cols-[5fr_5fr] gap-8">
<div>

### Local connectivity
* An output pixel depends only on a **small region** of the input: its **receptive field**
* Deeper layers have larger receptive fields

### Weight sharing
* The **same kernel** is applied at all positions
* MLP: $10^9$ parameters for a 1 MP image;<br> CNN: $3 \times 3 \times c_{in} \times c_{out}$ ≈ a few thousand

### Pattern detectors
* The stronger the match with the kernel, the larger the output

</div>
<div>

### Translation equivariance
* Shift the input → the output shifts **the same way**
* A pattern is detected wherever it appears
* **Not** invariance yet: *where* it was detected still changes — pooling (later) adds a little invariance

<br>

> *"Convolution leverages three important ideas...: sparse interactions, parameter sharing and equivariant representations."*
> <small>— Goodfellow, Bengio & Courville, [Deep Learning, §9.2](https://www.deeplearningbook.org/contents/convnets.html)</small>

</div>
</div>

<span class="refs">Read: [CS231n, Convolutional Networks](https://cs231n.github.io/convolutional-networks/) · [d2l.ai 7.2](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html) · Watch: [A. Ng, C4W1L11 Why Convolutions](https://www.youtube.com/watch?v=ay3zYUeuyhU)</span>

<!--
**Presenter Notes:**

**Summary slide** - emphasize the key properties:

1. **Local connectivity**: Each output depends on a small input region
   - This is the "locality" principle in action
   - Reduces parameters dramatically

2. **Weight sharing**: Same kernel everywhere
   - This is why we get translation equivariance
   - A pattern detector works at any location

3. **Translation equivariance**: Shift input → shift output
   - Not quite invariance (that comes from pooling)
   - But pattern detection works anywhere
   - Demo in a few minutes: "shift input" in the Convolution Playground

4. **Hierarchical features**:
   - Early layers: simple patterns (edges)
   - Deep layers: complex patterns (objects)

Ng's "Why Convolutions" names the same two ideas: parameter sharing and sparsity of connections.
-->
