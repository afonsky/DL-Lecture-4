---
layout: center
---

<center>

# Pooling Layer

# Downsampling, and a little invariance
</center>

---
zoom: 0.95
---

# Pooling Layer
<div></div>

The pooling layer (**POOL**) performs **downsampling**, reducing spatial dimensions while retaining important information:

<div class="grid grid-cols-[2fr_1fr] gap-3">
<div>

### Max Pooling (most common)
* Selects **maximum value** in each window
* Preserves strongest activations (detected features)
* Provides some **translation invariance**
* *"If the feature was detected anywhere in the window, keep it"*
</div>
<div>
  <figure>
    <img src="/max-pooling-a.png" style="width: 230px !important;">
  </figure>  
</div>
</div>

<div class="grid grid-cols-[2fr_1fr]">
<div>

### Average Pooling
* Computes **mean** of values in the window
* Smoother downsampling
* Used in LeNet; modern networks: **global average pooling** before the classifier *(Lecture 5)*
</div>
<div>
  <figure>
    <img src="/average-pooling-a.png" style="width: 230px !important;">
  </figure>
</div>
</div>

<span class="refs">Images: [Stanford CS230 cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks) · Watch: [A. Ng, C4W1L09 Pooling Layers](https://www.youtube.com/watch?v=8oOgPUO-TBY)</span>

<!--
**Presenter Notes:**

**Two types of pooling:**

1. **Max Pooling** (most common):
   - Takes the maximum value in each window
   - Intuition: "Was the feature detected ANYWHERE in this region?"
   - Provides translation invariance within the window
   - Preserves the strongest activations

2. **Average Pooling**:
   - Takes the mean of values in the window
   - Smoother, less aggressive
   - Used in LeNet (historical) and at the end of modern networks
   - Global Average Pooling: average over entire feature map

**Key insight:** Pooling makes the representation more compact and adds invariance to small translations.
-->

---
zoom: 0.82
---

# Pooling: Properties and Purpose

<div class="grid grid-cols-[1fr_1fr] gap-10">
<div>

### What pooling does
* **Aggregates** information over a spatial window
* Applies to **each channel separately** (channels unchanged)
* **No parameters** — nothing to learn
* Typical: $2\times 2$ window, stride $2$ → halves height and width

### Why use pooling?
1. **Less computation** in the layers that follow
2. **Grows the receptive field** quickly
3. **Small shifts** of the input barely change the output
4. A **smaller fully connected head**: LeNet's has $400$ inputs, not $1\,600$

</div>
<div>

**Output size**, as for convolutions:
$$\lfloor (n + 2p - k)/s \rfloor + 1$$

```python
nn.MaxPool2d(kernel_size=2)   # stride defaults to kernel_size
nn.AvgPool2d(kernel_size=2)
```

### Modern trends
* **Strided convolutions** instead of pooling: the network learns its own downsampling ([Springenberg et al., 2014](https://arxiv.org/abs/1412.6806))
* **Global average pooling** replaces the fully connected head *(Lecture 5)*
* *"Discarding pooling layers has also been found to be important in training good generative models..."* <small>— [CS231n](https://cs231n.github.io/convolutional-networks/#pooling-layer)</small>

</div>
</div>

<span class="refs">Read: [d2l.ai 7.5](https://d2l.ai/chapter_convolutional-neural-networks/pooling.html) · [Goodfellow, Bengio & Courville, Deep Learning §9.3](https://www.deeplearningbook.org/contents/convnets.html)</span>

<!--
**Presenter Notes:**

**Why pooling matters:**

1. **Reduces computation**: Smaller feature maps = fewer operations
2. **Increases receptive field**: after a 2x2/stride-2 pool, every later layer's kernel covers twice as much of the input
3. **Translation invariance**: Small shifts don't change the max (next slide)
4. **Smaller head**: LeNet's last pool turns 16x10x10 = 1600 values into 16x5x5 = 400 before the first Linear layer

**Output size formula:** Same logic as convolution
- 2×2 pooling with stride 2 halves spatial dimensions
- 4× reduction in feature map size

**Modern debate:**
- Some architectures (All-CNN) remove pooling entirely
- Use strided convolutions instead
- Allows network to learn its own downsampling
- For classification: Global Average Pooling is now standard

**Important:** Pooling has NO learnable parameters!
-->

---
zoom: 0.95
---

# Equivariance vs. Invariance

<figure>
  <img src="/equivariance_pooling.svg" style="width: 900px !important; margin: 0 auto;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px; text-align: center">
    After Goodfellow, Bengio &amp; Courville, <i>Deep Learning</i>, Fig. 9.8. Orange outline: values that changed after the shift.
  </figcaption>
</figure>

<div class="grid grid-cols-2 gap-10">
<div>

* **Convolution is equivariant**: shift the input by one pixel and **every** detector output moves with it
* **Max-pooling is approximately invariant**: the max ignores *where* in its window the feature was — 4 of 7 outputs stay the same

</div>
<div>

<v-click>

* Only **approximately**, and only for **small** shifts: strided downsampling can still flip a prediction ([Azulay & Weiss, 2019](https://arxiv.org/abs/1805.12177); [Zhang, 2019](https://arxiv.org/abs/1904.11486))
* LeCun's LeNet-5 demos: [translation](http://yann.lecun.com/exdb/lenet/translation.html), [scale](http://yann.lecun.com/exdb/lenet/scale.html), [rotation](http://yann.lecun.com/exdb/lenet/rotation.html)

</v-click>
</div>
</div>

<span class="refs">*"...pooling helps to make the representation approximately invariant to small translations of the input."* — [Deep Learning, §9.3](https://www.deeplearningbook.org/contents/convnets.html)</span>

<!--
Read the figure row by row. Top rows: outputs of a detector (convolution + ReLU); after the shift every value is different,
but it is the same row moved one step - that is equivariance. Bottom rows: max over windows of width 3, stride 1.
Only 3 of the 7 pooled values changed, because each window only asks "is there a strong response somewhere in here?".
With a 2x2 window and stride 2 the story is similar but cruder: a feature that moves within its window is invisible to
the next layer, one that crosses a window boundary is not - which is why CNNs are not truly shift-invariant (Zhang 2019).
We will measure this on real images in a moment (LeNet vs. MLP).
-->
