---
layout: center
---

# Convolutional Layer

---

# Convolutional Layer

### This scan-like approach is realized in **convolution layers** of NNs:
<br>
<div class="grid grid-cols-[5fr_3fr] gap-4">
<div>
<center>
  <figure>
    <img src="/conv_2D_1.gif" style="width: 450px !important;">
  </figure>
</center>
<br>
<br>

### Key insight:
> *"The same kernel is applied everywhere — this is called **weight sharing**"*
</div>
<div>
<br>

### Key terminology:
* <span style="color: #268BD2">**Input:**</span> $(5\times 5)$
* <span style="color: #1A6998">**Kernel/Filter**</span>: $(3\times 3)$
  * Contains **learnable weights**
* <span style="color: #2AA098">**Output/Feature Map**</span>: $(3\times 3)$
  * Also called **activation map**
</div>
</div>

<br>
<span style="color:grey"><small>Gifs are from <a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning</a> by Vincent Dumoulin and Francesco Visin</small></span>

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
3. Sum all products to get ONE output value
4. Slide the kernel and repeat

**Weight sharing insight:** The SAME kernel weights are used at EVERY position. This is what gives us translation equivariance and dramatically reduces parameters.
-->

---

# Convolutional Layer (1D)
<div>
<center>
  <figure>
    <img src="/conv_1D_1.gif" style="width: 350px !important;">
  </figure>
</center>   
</div>

#### A convolution is an operation between two signals: input and kernel.

#### To get the convolution output of an input vector and a kernel:
- **Slide the kernel** at each different possible positions in the input
- For each position, perform the **element-wise product**<br> between the kernel and the corresponding part of the input
- **Sum** the result of the element-wise product

<br>
<span style="color:grey"><small>Gifs are from <a href="https://arxiv.org/abs/1603.07285">A guide to convolution arithmetic for deep learning</a> by Vincent Dumoulin and Francesco Visin</small></span>

<!--
**Presenter Notes:**

Start with 1D to build intuition before moving to 2D:

**The three steps:**
1. **Slide**: Move the kernel across the input
2. **Multiply**: Element-wise multiplication at each position
3. **Sum**: Add up all products to get one output value

**Example on board:**
- Input: [1, 2, 3, 4, 5]
- Kernel: [1, 0, -1]
- First output: 1×1 + 2×0 + 3×(-1) = -2

This kernel computes a **discrete derivative** - it detects changes/edges!
-->

---

# Cross-Correlation vs Convolution
<div></div>

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

### Strictly speaking...
* What we call "convolution" in deep learning is actually **cross-correlation**
* True convolution requires **flipping the kernel** horizontally and vertically

<br>

### Why does this matter?
* It **doesn't** in practice!
* Since kernels are **learned from data**, a learned kernel is simply the flipped version of what true convolution would learn
* The outputs remain **identical**

</div>
<div>
<br>
<br>

### Cross-correlation (what we use):
$$Y[i,j] = \sum_a \sum_b K[a,b] \cdot X[i+a, j+b]$$

### True convolution:
$$Y[i,j] = \sum_a \sum_b K[a,b] \cdot X[i-a, j-b]$$

<br>

> *"We follow the standard terminology in deep learning and refer to cross-correlation as convolution."*
> — d2l.ai

</div>
</div>

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

---

# Convolutional Layer Parameters
<div></div>

Assuming the given input $I$ with respect to its dimensions, we can select the following
**hyperparameters** of the convolution layer:
* Kernel (**Filter**) size $F$
* **Stride** $S$ - the number of *pixels* by which the window moves after each operation
* **Zero-padding** $P$ - the number of zeroes at each side of the boundaries of the input

<div class="grid grid-cols-[5fr_2fr]">
<div>
<center>
  <figure>
    <img src="/conv_layer_1.png" style="width: 450px !important;">
  </figure>
</center>   
</div>
<div>
<br>

#### Therefore:
<br> $\boxed{O = \frac{I - F + 2P}{S} + 1}$
</div>
</div>

<span style="color:grey"><small>Images are from <a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks">Convolutional Neural Networks cheatsheet</a></small></span>

<!--
**Presenter Notes:**

**Three key hyperparameters:**

1. **Filter size (F)**: Typically 3×3 or 5×5
   - Larger = bigger receptive field but more parameters
   - Modern trend: stack small filters (VGG insight)

2. **Stride (S)**: How much to move the kernel
   - Stride 1: Move 1 pixel at a time (default)
   - Stride 2: Skip every other position (downsamples)

3. **Padding (P)**: Add zeros around the border
   - "Same" padding: Output size = Input size
   - "Valid" padding: No padding, output shrinks

**Formula derivation:** Work through on the board:
- Input 5×5, Filter 3×3, Stride 1, Padding 0
- Output = (5 - 3 + 0)/1 + 1 = 3
-->

---

# Convolutional Layer: Multiple Channels
<div></div>

You can use have : multiple channels in input. A color image usually have 3 input channels: RGB.
Therefore, the kernel will also have channels, one for each input channel.

<div class="grid grid-cols-[5fr_3fr] gap-8">
<div>
  <figure>
    <img src="/conv-2d-in-channels.gif" style="width: 550px !important;">
  </figure>  
</div>
<div>
  <figure>
    <img src="/conv-2d-out-channels.gif" style="width: 400px !important;">
  </figure>
</div>
</div>

<br>
<span style="color:grey"><small>Gifs are from <a href="https://github.com/theevann/amld-pytorch-workshop/blob/master/6-CNN.ipynb">the PyTorch Workshop at Applied ML Days 2019</a></small></span>

<!--
**Presenter Notes:**

**Channels are crucial** - make sure students understand:

**Input channels:**
- RGB image = 3 channels
- Kernel has shape: (height × width × input_channels)
- Example: 3×3×3 = 27 weights for RGB

**Output channels:**
- Each output channel uses a DIFFERENT kernel
- Multiple kernels = multiple feature detectors
- One might detect vertical edges, another horizontal

**Dimensions:**
- Input: H × W × C_in
- Kernel: K × K × C_in × C_out  
- Output: H' × W' × C_out

**Total parameters:** K × K × C_in × C_out + C_out (bias)
-->

---
layout: center
---

# Images as Functions

---

# Images as Functions

### No change:
<br>
  <figure>
    <img src="/im_no_change.gif" style="width: 500px !important;">
  </figure>
<br>
<br>
<br>
<br>
<br>
<br>
<span style="color:grey"><small>Gifs are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Images as Functions

### Shifted right by one pixel:
<br>
  <figure>
    <img src="/im_1_pixel_right.gif" style="width: 500px !important;">
  </figure>
<br>
<br>
<br>
<br>
<br>
<br>
<span style="color:grey"><small>Gifs are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Images as Functions

### Blurred:
<br>
  <figure>
    <img src="/im_blurred.gif" style="width: 500px !important;">
  </figure>
<br>
<br>
<br>
<br>
<br>
<br>
<span style="color:grey"><small>Gifs are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Images as Functions

### How to obtain sharpened image?

Step 1: Original - Smoothed = "Details"
<br>
  <figure>
    <img src="/sharpening1.png" style="width: 570px !important;">
  </figure>
<br>


<span style="color:grey"><small>Images are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Images as Functions

### How to obtain sharpened image?

Step 2: Original + "Details" = Sharpened
<br>
  <figure>
    <img src="/sharpening2.png" style="width: 590px !important;">
  </figure>
<br>


<span style="color:grey"><small>Images are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Images as Functions

### Sharpened:
<br>
  <figure>
    <img src="/im_sharpened.gif" style="width: 570px !important;">
  </figure>
<br>
<br>
<br>
<br>
<br>
<br>
<span style="color:grey"><small>Gifs are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

---

# Image Kernels Explained Visually <a href="https://setosa.io/ev/image-kernels/">[link]</a>

<iframe src="https://setosa.io/ev/image-kernels/" width="1100" height="550" style="-webkit-transform:scale(0.8);-moz-transform-scale(0.8); position: relative; top: -65px; left: -120px"></iframe>

---

# Convolutions: Key Properties

<div class="grid grid-cols-[5fr_5fr] gap-6">
<div>

### Local Connectivity
* A pixel in the output depends only on a **small region** of the input
* This region is called the **receptive field**
* Deeper layers have larger receptive fields

### Weight Sharing
* The **same kernel** is applied at all positions
* Dramatically reduces parameters:
  * MLP: $10^9$ parameters for 1MP image
  * CNN: $3 \times 3 \times c_{in} \times c_{out}$<br> ≈ few thousand

</div>
<div>

### Translation Equivariance
* If input shifts, output shifts accordingly
* Pattern detection works regardless of location

### What kernels detect
* Kernels act as **pattern detectors**
* The stronger the match, the larger the output value
* Early layers: edges, textures
* Deeper layers: parts, objects

> *"ConvNets learn a hierarchy of increasingly complex features."*<br> — Andrej Karpathy (CS231n)

</div>
</div>

<span style="color:grey"><small>Based on <a href="http://cs231n.stanford.edu/">Stanford CS231n</a> by Andrej Karpathy and <a href="https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html">d2l.ai</a></small></span>

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

4. **Hierarchical features**: 
   - Early layers: simple patterns (edges)
   - Deep layers: complex patterns (objects)
   - This is the "deep" in deep learning!

**Karpathy quote** summarizes the key insight.
-->