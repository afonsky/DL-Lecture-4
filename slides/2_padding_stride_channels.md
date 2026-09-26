---
layout: center
---

<center>

# Padding, Stride and Channels

# Controlling the shape of the output
</center>

---
zoom: 0.93
---

# Padding: Keeping the Border

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Without padding, every layer shrinks the map
* A $k\times k$ kernel turns $n\times n$ into $(n-k+1)\times(n-k+1)$
* $240\times240$ through ten $5\times5$ layers → $200\times200$: **30 %** of the pixels gone
* Corner pixels are used by just **one** window

<figure>
  <img src="/conv-reuse.svg" style="width: 330px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
    <a href="https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html">d2l.ai Fig. 7.3.1 Pixel utilization for convolutions of size 1×1, 2×2, and 3×3 respectively</a>
  </figcaption>
</figure>

</div>
<div>

### Zero padding
* Add $p$ rows and columns of zeros on **each** side
* **"Same" padding** $p = (k-1)/2$ keeps the size — one reason kernels are odd: $1, 3, 5, 7$

<figure>
  <img src="/conv-pad.svg" style="width: 330px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
    <a href="https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html">d2l.ai Fig. 7.3.2 Two-dimensional cross-correlation with padding</a>
  </figcaption>
</figure>

```python
nn.Conv2d(1, 1, kernel_size=3, padding=1)       # 8x8 -> 8x8
nn.Conv2d(1, 1, kernel_size=5, padding="same")  # size kept
```

</div>
</div>

<span class="refs">Read: [d2l.ai 7.3.1](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html#padding) · Watch: [A. Ng, C4W1L04 Padding](https://www.youtube.com/watch?v=smHa2442Ah4)</span>

<!--
Left figure: numbers are how many windows use each pixel; the corners of a 3x3 convolution are used once, the centre nine times.
Right figure: a 3x3 input padded to 5x5, a 2x2 kernel, a 4x4 output - one row and one column larger than the input.
Side remark from d2l: zero padding lets the network find the image border, so CNNs can learn some absolute position information.
padding="same" works for stride 1 only.
-->

---
zoom: 0.85
---

# Stride and the Output Size

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Stride $s$: move the window $s$ pixels at a time
* Skips positions → downsamples by $\approx s$ in each direction
* Cheaper than computing every position and discarding most

<figure>
  <img src="/conv-stride.svg" style="width: 340px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
    <a href="https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html">d2l.ai Fig. 7.3.3 Cross-correlation with strides of 3 and 2 for height and width, respectively</a>
  </figcaption>
</figure>

</div>
<div>

### The formula you will use every week
$$\boxed{\;n_\text{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1\;}$$
<small>$n$ input size, $k$ = `kernel_size`, $p$ = `padding` (per side), $s$ = `stride`</small>

<v-click>
<div class="compact-table">

| $n$ | $k$ | $p$ | $s$ | $n_\text{out}$ | |
|---|---|---|---|---|---|
| 28 | 5 | 2 | 1 | **28** | LeNet, layer 1 |
| 32 | 3 | 1 | 2 | **16** | halves the size |
| 227 | 11 | 0 | 4 | **55** | AlexNet (Lecture 5) |
| 7 | 3 | 0 | 3 | **2** | last column unused |

</div>
</v-click>

</div>
</div>

<style>
.compact-table table { font-size: 0.85em; }
.compact-table td, .compact-table th { padding-top: 0.2em; padding-bottom: 0.2em; }
</style>

<span class="refs">Read: [d2l.ai 7.3.2](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html#stride) · [CS231n, "Spatial arrangement"](https://cs231n.github.io/convolutional-networks/#convolutional-layer) · [Dumoulin & Visin, conv arithmetic animations](https://github.com/vdumoulin/conv_arithmetic) · Watch: [A. Ng, C4W1L05 Strided Convolutions](https://www.youtube.com/watch?v=tQYZaDn_kSg)</span>

<!--
The figure uses d2l's convention (padding 1 on a 3x3 input, stride 3 down and 2 across): (3 + 2 - 2)/3 + 1 = 2 and floor((3 + 2 - 2)/2) + 1 = 2.
CS231n writes the same formula as (W - F + 2P)/S + 1 and asks for it to be an integer; PyTorch floors it silently.
Let the class compute the first two examples before clicking.
Height and width can use different k, p, s: apply the formula to each.
-->

---
zoom: 0.9
---

# Try It: Convolution Playground

<ConvPlayground />

<div class="mt-3" style="font-size: 0.72em">

**Hover** an output cell: its window (orange) × kernel = products → Σ &nbsp;·&nbsp; change **padding** and **stride** and check the formula &nbsp;·&nbsp; press **shift input →**: the output pattern moves with the input — **equivariance**

</div>

<!--
Live demo, 2-3 min.
1. Vertical edges on the bar: the output is -3 on the left edge, +3 on the right edge, 0 inside - "silent on flat regions".
   Hover a -3 cell and read the products: the left column of the window is 0, the right column is 1.
2. Horizontal edges on the bar: all zeros except the top and bottom rows - the bar has no horizontal edges there.
3. padding 1: output 7x7 = input size ("same"); the new border cells see the dashed zeros.
4. stride 2: output 3x3 (padding 0) - check with the formula on the slide.
5. Back to stride 1, padding 0, "shift input" twice: the -3/+3 columns move right with the bar. That is equivariance.
   With stride 2, shift once: the output does NOT simply shift - a preview of why strided layers are only approximately shift-invariant.
6. Optional: draw a diagonal with clicks and compare the edge kernels.
-->

---
zoom: 0.84
---

# Convolutional Layer: Multiple Channels

A colour image has **3 input channels** (RGB), so each kernel has **one $k\times k$ slice per input channel**. The slices' results are **summed** (plus one bias) into **one** output channel; **$c_\text{out}$ kernels give $c_\text{out}$ output channels**.

<div class="grid grid-cols-[5fr_3fr] gap-8">
<div>
  <figure>
    <img src="/conv-2d-in-channels.gif" style="width: 460px !important;">
  </figure>  
</div>
<div>
  <figure>
    <img src="/conv-2d-out-channels.gif" style="width: 300px !important;">
  </figure>
</div>
</div>

<div class="grid grid-cols-[1fr_1fr] gap-8">
<div>

**Shapes:** input $c_\text{in}\times h \times w$<br> → kernel tensor $c_\text{out}\times c_\text{in}\times k\times k$<br> → output $c_\text{out}\times h'\times w'$

</div>
<div>

```python
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
conv.weight.shape   # torch.Size([16, 3, 3, 3])
conv.bias.shape     # torch.Size([16])
```

</div>
</div>

<span class="refs">GIFs: [AMLD 2019 PyTorch workshop](https://github.com/theevann/amld-pytorch-workshop/blob/master/6-CNN.ipynb) · Read: [d2l.ai 7.4](https://d2l.ai/chapter_convolutional-neural-networks/channels.html) · Watch: [A. Ng, C4W1L06 Convolutions Over Volume](https://www.youtube.com/watch?v=KTB_OFoAQcc) · [C4W1L07 One Layer of a Convolutional Net](https://www.youtube.com/watch?v=jPOAS7uCODQ)</span>

<!--
**Presenter Notes:**

**Channels are crucial** - make sure students understand:

**Input channels:**
- RGB image = 3 channels
- Kernel has shape: (input_channels × height × width)
- Example: 3×3×3 = 27 weights for RGB
- Left GIF: 308 + (-498) + 164 + bias 1 = -25, one number of ONE output channel

**Output channels:**
- Each output channel uses a DIFFERENT kernel
- Multiple kernels = multiple feature detectors
- One might detect vertical edges, another horizontal
- d2l's caveat: channels are learned to be jointly useful, not one clean detector each

**Dimensions (PyTorch order):**
- Input: N × C_in × H × W
- Kernel: C_out × C_in × K × K
- Output: N × C_out × H' × W'

**Total parameters:** K × K × C_in × C_out + C_out (bias) = 3·3·3·16 + 16 = 448 for the code on the slide
-->

---
zoom: 0.95
---

# 1×1 Convolutions

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>
<br>
  <figure>
    <img src="/conv-1x1.svg" style="width: 420px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
      <a href="https://d2l.ai/chapter_convolutional-neural-networks/channels.html">d2l.ai Fig. 7.4.2 The cross-correlation computation uses the 1×1 convolution kernel with three input channels and two output channels. The input and output have the same height and width</a>
    </figcaption>
  </figure>
</div>
<div>

<v-clicks>

* Sees **one pixel** — but **all its channels**
* = a fully connected layer $c_\text{in} \to c_\text{out}$, applied at **every pixel** with shared weights
* $c_\text{in} c_\text{out} + c_\text{out}$ parameters; height and width unchanged
* The cheap way to **change the number of channels**: `nn.Conv2d(256, 64, kernel_size=1)` has $16{,}448$ parameters
* Everywhere in Lecture 5: Network in Network, GoogLeNet, ResNet bottlenecks

</v-clicks>

</div>
</div>

<span class="refs">Read: [d2l.ai 7.4.3](https://d2l.ai/chapter_convolutional-neural-networks/channels.html#times-1-convolutional-layer) · [Lin, Chen & Yan, Network in Network (2013)](https://arxiv.org/abs/1312.4400) · Watch: [A. Ng, C4W2L05 Network in Network](https://www.youtube.com/watch?v=c1RBQzKsDCk)</span>

<!--
The figure: each output pixel is a weighted sum of the 3 input values at the same location - a 3 -> 2 linear layer.
Followed by a ReLU, a 1x1 conv is a per-pixel MLP layer; that is exactly the "network in network" idea.
Ask: 256 channels in, 64 out, 1x1 kernel: how many weights? 256 x 64 + 64 = 16,448.
The same map with a 3x3 kernel would need 9x as many weights - which is why bottleneck blocks squeeze channels with 1x1 first.
-->

---
zoom: 0.86
---

# Counting Parameters and Compute

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Parameters: independent of the image size
$$\#\text{params} = (k^2\, c_\text{in} + 1)\, c_\text{out}$$

* `nn.Conv2d(3, 64, 3)` on $224\times224$ RGB: $(9\cdot3+1)\cdot64 = 1{,}792$
* A fully connected layer between the same input and output maps: $\approx 4.8\times10^{11}$

<v-click>

### Karpathy's example *(CS231n)*
AlexNet's first layer: $55\cdot55\cdot96 = 290{,}400$ units, $11\cdot11\cdot3 + 1 = 364$ parameters each. **105.7 M** without sharing, **34,944** with it

</v-click>
</div>
<div>

### Compute: grows with the image
$$\#\text{multiply-adds} = h_\text{out}\, w_\text{out}\, c_\text{out} \cdot k^2 c_\text{in}$$

<v-clicks>

* Every weight is reused at **every output position**
* $256\times256$ image, $5\times5$ kernel, $128\to128$ channels: **26.8 G** multiply-adds, **one** layer *(d2l 7.4.4)*
* Few parameters, much arithmetic: a CNN can be **slower** than an MLP with more parameters

</v-clicks>

```python
sum(p.numel() for p in model.parameters())   # count them
```

</div>
</div>

<span class="refs">Read: [CS231n, "Real-world example"](https://cs231n.github.io/convolutional-networks/#convolutional-layer) · [d2l.ai 7.4.4 Discussion](https://d2l.ai/chapter_convolutional-neural-networks/channels.html#discussion) · Watch: [A. Ng, C4W1L08 Simple Convolutional Network Example](https://www.youtube.com/watch?v=3PyJA9AfwSk)</span>

<!--
Fully connected: 224·224·3 = 150,528 inputs times 224·224·64 = 3,211,264 outputs = 4.8 x 10^11 weights.
CS231n: 290,400 x 364 = 105,705,600 without sharing; 96 x 364 = 34,944 with sharing.
d2l counts multiplications and additions separately and says "over 53 billion operations": 2 x 26.8 G.
Our LeNet (next section) has 3.3x fewer parameters than Lecture 3's MLP but needs about 2x more multiply-adds per image.
Ask: what happens to parameters and compute if the image doubles in size? Parameters: nothing. Compute: x4.
-->
