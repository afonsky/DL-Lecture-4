---
layout: center
---

# Backup Slides

---
zoom: 0.8
---

# How Frameworks Compute a Convolution

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### Forward: one big matrix product (im2col)
* Stretch every $k\times k\times c_\text{in}$ window into a **column**: a matrix $X_\text{col}$ of shape $(k^2 c_\text{in}) \times (h_\text{out} w_\text{out})$
* Stretch every kernel into a **row**: $W_\text{row}$ of shape $c_\text{out} \times (k^2 c_\text{in})$
* $W_\text{row} X_\text{col}$ is the whole layer: $c_\text{out}\times(h_\text{out}w_\text{out})$, reshaped to $c_\text{out}\times h_\text{out}\times w_\text{out}$
* CS231n's AlexNet example: $X_\text{col}$ is $363\times3025$, $W_\text{row}$ is $96\times363$

</div>
<div>

### The trade-offs
* Overlapping windows are **copied** — up to $k^2$ times more memory
* In exchange, the layer runs on the most optimized routine there is: matrix multiplication (BLAS, cuBLAS)
* cuDNN also has FFT and Winograd algorithms and picks the fastest per layer: `torch.backends.cudnn.benchmark = True`

### Backward
* *"The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters)."* <small>— CS231n</small>
* Autograd does it for you *(Lecture 3)*

</div>
</div>

<span class="refs">Read: [CS231n, "Implementation as Matrix Multiplication"](https://cs231n.github.io/convolutional-networks/#convolutional-layer) · [d2l.ai 7.2 Exercise 4](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html#exercises) · [Chetlur et al., cuDNN (2014)](https://arxiv.org/abs/1410.0759)</span>
