---
layout: center
---

# Pooling Layer

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
    <img src="/max-pooling-a.png" style="width: 250px !important;">
  </figure>  
</div>
</div>

<div class="grid grid-cols-[2fr_1fr]">
<div>
<br>

### Average Pooling
* Computes **mean** of values in the window
* Smoother downsampling
* Used in LeNet, often at final layers
* Modern networks: **Global Average Pooling** before classifier
</div>
<div>
  <figure>
    <img src="/average-pooling-a.png" style="width: 250px !important;">
  </figure>

<span style="color:grey"><small>Gifs are from <a href="https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks">Stanford CS230 Cheatsheet</a></small></span>
</div>
</div>

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
zoom: 0.9
---

# Pooling: Properties and Purpose
<br>

<div class="grid grid-cols-[1fr_1fr] gap-6">
<div>

### What pooling does
* **Aggregates** information over a spatial window
* Applies to **each channel separately**<br> (channels unchanged)
* Typical window: $2\times 2$ with stride $2$<br> → quarters spatial size

<br>

### Why use pooling?
1. **Reduces computation** in subsequent layers
2. **Increases receptive field** efficiently
3. **Translation invariance**: small shifts in input don't affect output
4. **Reduces overfitting** by providing abstraction

</div>
<div>

For input size $I$, pooling size $F$, stride $S$:<br>
Output size: $\boxed{O = \frac{I - F}{S} + 1}$

### Modern trends (Karpathy, CS231n)
* Some architectures use **strided convolutions** instead of pooling
* **Global Average Pooling** replaces fully-connected layers
* Debate: Is pooling still necessary?
  * *"Getting rid of pooling layers... may prove important for training generative models"*<br> — Springenberg et al. (see [arXiv:1412.6806](https://arxiv.org/abs/1412.6806))

</div>
</div>

<span style="color:grey"><small>See also: <a href="https://d2l.ai/chapter_convolutional-neural-networks/pooling.html">d2l.ai Ch. 7.5</a></small></span>

<!--
**Presenter Notes:**

**Why pooling matters:**

1. **Reduces computation**: Smaller feature maps = fewer operations
2. **Increases receptive field**: 2×2 pooling doubles the effective receptive field
3. **Translation invariance**: Small shifts don't change the max
4. **Regularization**: Forces the network to learn more robust features

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