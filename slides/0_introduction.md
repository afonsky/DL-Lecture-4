# Essentials of Artificial Neural Networks

### Building blocks (<span style="color:#FA9370">new</span>):
<div class="grid grid-cols-[3fr_2fr_2fr] gap-3">
<div>

* Neuron
* Fully-connected (a.k.a. *Linear*) layer
* Activation function
* Recurrent layer (future lecture)
</div>

<div>
<v-clicks>

* Loss function
* <span style="color:#FA9370">**Convolution layer**</span>
* <span style="color:#FA9370">**Pooling layer**</span>
</v-clicks>
</div>

<div>
  <figure>
    <img src="/lego_A.jpg" style="width: 200px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Image source:
      <a href="http://sgaguilarmjargueso.blogspot.com/2014/08/de-lego.html">http://sgaguilarmjargueso.blogspot.com</a>
    </figcaption>
  </figure>   
</div>
</div>

### Key concepts (review from previous lectures | <span style="color:#FA9370">new</span>):
<div class="grid grid-cols-[1fr_1fr_1fr] gap-3">
<div>

* Weights & Biases
* Backpropagation
* Gradient descent

</div>
<div>

* Learning rate
* MiniBatch
* Regularization
</div>

<div>
<v-click at="4">

* <span style="color:#FA9370">**Translation invariance**</span>
</v-click>
<v-click at="5">

* <span style="color:#FA9370">**Locality**</span>
</v-click>
<v-click at="6">

* <span style="color:#FA9370">**Weight sharing**</span>
</v-click>
</div>

</div>

<!--
**Presenter Notes:**

Today we're introducing two new building blocks: **Convolution layers** and **Pooling layers**.

Key points to emphasize:
- Students already know the foundational concepts (weights, backprop, gradient descent)
- Today we add **three new concepts**: translation invariance, locality, and weight sharing
- These concepts are what make CNNs so powerful for image data
- Convolution + Pooling are the "secret sauce" that allows us to process images efficiently

**Transition:** Let's start by understanding why we need something different from fully-connected networks for images.
-->

---
layout: center
---

# Fully-Connected NNs Limitations

---

# <center>Neural networks so far</center>
<br>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_1.png" style="width: 550px !important;">
  </figure>
</center>
</div>
<br>
<br>

# <center>Can recognize patterns in data (e.g. digits)</center>
<span style="color:grey"><small>The images in the slides 3-9 are from <a href="https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html" target="_blank">https://ai.stanford.edu/~syyeung/cvweb/tutorial1.html</a></small></span>

<!--
**Presenter Notes:**

Start with what students already know:
- We've seen that neural networks can recognize patterns like handwritten digits
- The weights in a neural network act as **pattern templates**
- When the input matches the pattern, we get a high activation

**Key question to pose:** But what happens if we want to recognize a digit that appears in a different location in the image?

**Transition:** Let's see why this is a problem for standard fully-connected networks.
-->

---
zoom: 0.95
---

# The Problem with Fully-Connected NNs for Images

<div class="grid grid-cols-[3fr_2fr] gap-4">
<div>

### Computational infeasibility
* A **1-megapixel** image has $10^6$ input dimensions
* Even with 1000 hidden units: $10^6 \times 10^3 = 10^9$ parameters!
* This is just for **one layer**

<br>

### Ignores spatial structure
* Images are **not** random collections of pixels
* MLPs treat images as flat vectors<br> — **permutation invariant**
* A cat in the top-left looks completely different<br> from a cat in the bottom-right
</div>
<div>
<br>

> *"The key insight is that for images, we can exploit structure to massively reduce parameters while improving generalization."*
> 
> — Yann LeCun

<br>
<br>

### No built-in priors
* We know nearby pixels are related
* We know patterns can appear anywhere
* **MLPs don't encode this knowledge**

<span style="color:grey"><small>Based on <a href="https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html">d2l.ai Ch. 7.1</a></small></span>
</div>
</div>

<!--
**Presenter Notes:**

This is a **crucial slide** - make sure students understand the scale of the problem:

1. **Computational infeasibility**: Do the math on the board
   - 1 megapixel = 1,000,000 pixels
   - 1000 hidden units = 1 billion parameters for ONE layer!
   - Modern images are much larger (4K = 8 megapixels)

2. **Spatial structure ignored**: 
   - If you flatten an image to a vector, pixel 1 and pixel 1000 look equally "close"
   - But in reality, neighboring pixels are highly correlated

3. **No built-in priors**:
   - We KNOW that a cat is a cat regardless of where it appears
   - MLPs have to learn this from scratch for every position

**Yann LeCun quote** emphasizes the key insight that will lead us to CNNs.
-->

---

# <center>The weights look for patterns</center>

<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_2.png" style="width: 620px !important;">
  </figure>
</center>   
</div>
<br>

### The green pattern looks more like the weights pattern (black) than the red pattern
* The green pattern is more *correlated* with the weights

<!--
**Presenter Notes:**

This slide builds intuition for how neural networks detect patterns:

- The **weights** define a pattern template
- **High correlation** between input and weights = high activation
- The green pattern is more similar to the weight pattern, so it produces a stronger response

**Connection to convolution:** This is exactly what a convolution does - it measures similarity between the kernel (weights) and each patch of the input.

**Ask students:** What happens if the pattern we're looking for appears in a different location?
-->

---

# <center>Flower</center>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_3.jpg" style="width: 650px !important;">
  </figure>
</center>    
</div>
<br>
<br>

# <center>Is there a flower in any of these images?</center>

---

# <center>Flower</center>
<br>

<div>
<center>
  <figure>
    <img src="/nn_patterns_4.jpg" style="width: 650px !important;">
  </figure>
</center>   
</div>
<br>

* Will a NN that recognizes the left image as a flower<br> also recognize the one on the right as a flower?
* Need a network that will “fire” regardless of the precise location of the target object
<!--
**Presenter Notes:**

This slide motivates **translation invariance**:

- A fully-connected network trained on centered flowers might **fail** on off-center flowers
- The network has learned weights for specific pixel positions
- Moving the flower means completely different input neurons are activated

**Key insight:** We need a network that can detect "flower-ness" regardless of WHERE the flower appears.

**Solution preview:** What if we used the SAME weights to scan across the entire image?
-->
---

# The Need for Translation Invariance
<br>

<div class="grid grid-cols-[5fr_2fr] gap-4">
<div>

### The problem
* In many problems the **location** of a pattern is not important
  * Only the **presence** of the pattern matters
* Conventional NNs are sensitive to location of pattern
  * Moving it by one pixel results in an entirely different input
</div>
<div>

> *"A good representation is one that makes a subsequent learning task easier. In images, the task of recognition should not depend on where the object is."*
>
> — Yann LeCun

</div>
</div>

<br>

### Two key principles:
1. **Translation invariance**: The network should respond similarly to the same patch, regardless of where it appears
2. **Locality**: Early layers should focus on local regions, without regard for distant parts of the image

<!--
**Presenter Notes:**

These are the **two fundamental principles** behind CNNs:

1. **Translation invariance** (or equivariance):
   - The same pattern should be detected regardless of position
   - In practice, CNNs are translation **equivariant**: if input shifts, output shifts too
   - True invariance comes from pooling layers

2. **Locality**:
   - To understand a pixel, we only need to look at nearby pixels
   - A pixel in the corner doesn't directly affect one in the center
   - This is a strong prior about natural images

**LeCun quote** captures the essence: good representations shouldn't depend on object location.
-->

---

# Solution: Scan

<div>
<center>
  <figure>
    <img src="/nn_patterns_5.jpg" style="width: 600px !important;">
  </figure>
</center>   
</div>

### Scan for the desired object
* “Look” for the target object at each position
* At each location, entire region is sent through NN
<!--
**Presenter Notes:**

This is the **key intuition** behind convolution:

- Instead of training separate detectors for each location...
- We use **ONE detector** and slide it across the image
- At each position, we check: "Is the pattern here?"

**Analogy:** Like using a magnifying glass to scan a document for a specific word.

**Critical point:** The same weights are used at every position - this is **weight sharing**.
-->
---

# Solution: Scan

<div>
<center>
  <figure>
    <img src="/nn_patterns_6.jpg" style="width: 550px !important;">
  </figure>
</center>   
</div>

### Determine if any of the locations had a flower
* Each neuron in the right represents the output of the NN when it classifies one location in the input figure
* Look at the maximum value
  * Or pass it through a simple NN (e.g. linear combination + softmax)