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

* <span style="color:#FA9370">**Locality**</span>
</v-click>
<v-click at="5">

* <span style="color:#FA9370">**Weight sharing**</span>
</v-click>
<v-click at="6">

* <span style="color:#FA9370">**Translation equivariance**</span>
</v-click>
</div>

</div>

<!--
**Presenter Notes:**

Today we're introducing two new building blocks: **Convolution layers** and **Pooling layers**.

Key points to emphasize:
- Students already know the foundational concepts (weights, backprop, gradient descent)
- Today we add **three new concepts**: locality, weight sharing and translation equivariance (with pooling, approximate invariance)
- These concepts are what make CNNs so powerful for image data
- Convolution + Pooling are the "secret sauce" that allows us to process images efficiently

**Transition:** Let's start by understanding why we need something different from fully-connected networks for images.
-->

---
zoom: 0.85
---

# From Lecture 3 to Today

<div class="grid grid-cols-2 gap-16">
<div>

### We already have *(Lectures 1–3)*
* **forward → loss → backward → step** on minibatches
* MLPs with ReLU, He initialization, dropout
* Lecture 3's MLP for $28\times28$ images: $784 \to 256 \to 10$, **203,530** parameters

### We are still missing
* The MLP never learns **where** a pixel is: shuffle all pixels the same way and it trains just as well
* One megapixel × 1000 hidden units = $10^9$ weights in the first layer

</div>
<div>

### Plan for today
1. **Why not an MLP for images?** <small>locality, translation invariance</small>
2. **The convolution** <small>kernels, cross-correlation, learning a kernel</small>
3. **Padding, stride and channels** <small>output shapes, parameter counts</small>
4. **Pooling** <small>downsampling, a little invariance</small>
5. **LeNet in PyTorch** <small>vs. Lecture 3's MLP</small>
6. **What CNNs learn** <small>receptive fields, feature hierarchies</small>

</div>
</div>

<span class="refs">Main text: [d2l.ai, Ch. 7 Convolutional Neural Networks](https://d2l.ai/chapter_convolutional-neural-networks/index.html) · Next lecture: modern CNNs ([d2l.ai, Ch. 8](https://d2l.ai/chapter_convolutional-modern/index.html))</span>

<!--
Say, don't show: the training machinery of Lectures 1-3 does not change today; only the layers do.

Timing plan (80 min, of which 10-15 min is the quiz):
- Title, building blocks, this slide:           3 min
- 1. Why not an MLP for images:                 9 min
- 2. The convolution (incl. setosa demo):      10 min
- 3. Padding, stride, channels (playground):   10 min
- 4. Pooling:                                    4 min
- 5. LeNet in PyTorch (incl. MNIST demo):       10 min
- 6. What CNNs learn:                            5 min
- Conclusions:                                   3 min
                                       total   54 min  + quiz, ~10 min slack for questions

If running late, in this order:
  (a) skip "Convolutional Layer (1D)" and "Putting It All Together",
  (b) run the Convolution Playground for one minute only (vertical edges, then "shift input"),
  (c) skip the "Standard Recipe" slide - LeNet already shows the pattern.
-->

---
layout: center
---

<center>

# Fully Connected Networks Meet Images

# Why not just use an MLP?
</center>

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
<span style="color:grey"><small>Images in this section: B. Raj, CMU 11-785, <a href="https://deeplearning.cs.cmu.edu/F22/document/slides/lec9.CNN1.pdf" target="_blank">Scanning for patterns (aka convolutional networks)</a></small></span>

<!--
**Presenter Notes:**

Start with what students already know:
- We've seen that neural networks can recognize patterns like handwritten digits
- The weights in a neural network act as **pattern templates** (Lecture 2: the CIFAR-10 templates of a linear classifier)
- When the input matches the pattern, we get a high activation

**Key question to pose:** But what happens if we want to recognize a digit that appears in a different location in the image?

**Transition:** Let's see why this is a problem for standard fully-connected networks.
-->

---
zoom: 0.9
---

# The Problem with Fully-Connected NNs for Images

<div class="grid grid-cols-[3fr_2fr] gap-8">
<div>

### Computational infeasibility
* A **1-megapixel** image has $10^6$ input dimensions
* Even with 1000 hidden units: $10^6 \times 10^3 = 10^9$ parameters!
* This is just for **one layer**

### Ignores spatial structure
* Images are **not** random collections of pixels
* MLPs treat images as flat vectors — **permutation invariant**
* A cat in the top-left looks completely different to the MLP from a cat in the bottom-right

</div>
<div>

### What we know about images
> *"First, in array data such as images, local groups of values are often highly correlated, forming distinctive local motifs that are easily detected. Second, the local statistics of images and other signals are invariant to location."*
> <small>— LeCun, Bengio & Hinton, [Deep learning](https://www.nature.com/articles/nature14539), *Nature* (2015)</small>

### No built-in priors
* **MLPs don't encode this knowledge**: they must learn it anew for every position

</div>
</div>

<span class="refs">Read: [d2l.ai 7.1](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html) · Watch: [A. Ng, C4W1L01 Computer Vision](https://www.youtube.com/watch?v=ArPaAX_PhIs)</span>

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
   - We will measure this later today: an MLP trained on pixel-shuffled Fashion-MNIST is exactly as good as on real images

3. **No built-in priors**:
   - We KNOW that a cat is a cat regardless of where it appears
   - MLPs have to learn this from scratch for every position

The quote gives the two properties of images that the whole lecture exploits: local motifs (-> locality) and location-independent statistics (-> weight sharing).
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
zoom: 0.88
---

# The Need for Translation Invariance

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div>

### The problem
* Often only the **presence** of a pattern matters, not its **location**
* An MLP is sensitive to location: moving the pattern by one pixel gives an entirely different input

### Two key principles *(d2l 7.1)*
1. **Translation invariance**: early layers respond the same way to the same patch, **wherever** it appears
2. **Locality**: early layers look at **small regions**; deeper layers combine them

</div>
<div>
<br>
  <figure>
    <img src="/waldo-football.jpg" style="width: 360px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: relative; top: 6px">Image source:
      <a href="https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html">d2l.ai Fig. 7.1.1 Can you find Waldo (image courtesy of William Murphy (Infomatique))?</a>
    </figcaption>
  </figure>

<br>

*What Waldo looks like* does not depend on *where Waldo is*: sweep one **Waldo detector** over every patch.

</div>
</div>

<!--
**Presenter Notes:**

These are the **two fundamental principles** behind CNNs:

1. **Translation invariance** (or equivariance):
   - The same pattern should be detected regardless of position
   - In practice, CNNs are translation **equivariant**: if input shifts, output shifts too
   - Approximate invariance comes from pooling layers (later today)

2. **Locality**:
   - To understand a pixel, we only need to look at nearby pixels
   - A pixel in the corner doesn't directly affect one in the center
   - This is a strong prior about natural images

Waldo: give the class 10 seconds to find him. The point: a detector that scores every patch - many detection and segmentation systems work exactly like this.
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

---
zoom: 0.9
---

# From a Fully Connected Layer to a Convolution

<div class="grid grid-cols-[4fr_3fr] gap-8">
<div class="compact-table">

A $1000 \times 1000$ image → a hidden map of the same size:

| Assumption | Weights per hidden pixel | Parameters |
|---|---|---|
| none | one per input pixel | $10^{12}$ |
| + **translation invariance** | depend on the **offset** only | $4\times10^{6}$ |
| + **locality** | a $(2\Delta+1)^2$ **kernel** | $9$ ($\Delta = 1$) |

<v-click>

**Result: the convolutional layer** — one small kernel slid over the whole image, $10^{11}\times$ fewer parameters.

</v-click>
</div>
<div>

<v-click>

### A constrained fully connected layer
* Most weights forced to **zero** (locality), the rest forced to be **equal** (sharing)
* Abu-Mostafa's *hard constraint* (Learning From Data, Lecture 12): a smaller hypothesis set, fewer examples needed

</v-click>

<v-click>

> *"...we can think of the use of convolution as introducing an infinitely strong prior probability distribution over the parameters of a layer."*
> <small>— Goodfellow, Bengio & Courville, [Deep Learning, §9.4](https://www.deeplearningbook.org/contents/convnets.html)</small>

It pays off when the prior is true of the data — and costs accuracy when it is not.

</v-click>
</div>
</div>

<style>
.compact-table table { font-size: 0.8em; }
.compact-table td, .compact-table th { padding-top: 0.25em; padding-bottom: 0.25em; }
</style>

<span class="refs">Read: [d2l.ai 7.1.2](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html#constraining-the-mlp) · Watch: [Y. Abu-Mostafa, Lecture 12: Regularization](https://www.youtube.com/watch?v=I-VfYXzC5ro) · The full derivation is in the backup slides</span>

<!--
Walk the table top to bottom; this is d2l 7.1 without the index gymnastics (the derivation is in the backup).
Row 1: every one of the 10^6 hidden pixels has its own weight for each of the 10^6 input pixels.
Row 2: translation invariance says a shift of X should only shift H, so the weights may depend only
on the offset (a, b) between the two pixels, not on the position (i, j): a, b in (-1000, 1000) gives 4 x 10^6.
Row 3: locality zeroes every offset beyond Delta. Delta = 1 is the 3x3 kernel everyone uses today.
Eleven orders of magnitude, with no loss of resolution.

Abu-Mostafa link: in Lecture 12 he builds H2 from H10 by forcing w_q = 0 for q > 2 - a hard
constraint - and says a good regularizer constrains "in the direction of the target function".
Convolution is exactly such a constraint, and images satisfy it. We will measure what happens when
they do not (shuffled pixels) in the LeNet section.
-->
