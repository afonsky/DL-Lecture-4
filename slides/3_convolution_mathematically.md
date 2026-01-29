---
layout: center
---

# Math of Convolutional Layer

---
zoom: 0.92
---

# From Fully-Connected to Convolutions

<div class="grid grid-cols-[1fr_1fr] gap-4">
<div>

#### Starting from Multilayer Perceptron
* Let $\mathbf{X}$ be input image and $\mathbf{H}$ be hidden representation (same shape)
* Let $[\mathbf{X}]_{i, j}$ and $[\mathbf{H}]_{i, j}$ denote pixels at location $(i,j)$
* In a fully-connected layer:
$$[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}$$

</div>
<div>

#### The problem:
* For a $1000 \times 1000$ image: $10^{12}$ params!
* Each hidden unit connects to **every** pixel

<br>

#### What we want:
* A shift in input $\mathbf{X}$ should lead to the same shift in $\mathbf{H}$
</div>
</div>

#### Re-indexing using $k = i+a$, $l = j+b$:

$$[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b} [\mathbf{X}]_{i+a, j+b}$$

For translation invariance, $\mathsf{V}$ must not depend on $(i, j)$!

<!--
**Presenter Notes:**

**Mathematical derivation** of why convolutions make sense:

**Starting point:** Full MLP on images
- Every output pixel depends on EVERY input pixel
- 4th order weight tensor: W[i,j,k,l]
- For 1000×1000 image: 10^12 parameters!

**Re-indexing trick:**
- Instead of absolute positions (k,l), use relative offsets (a,b)
- k = i+a, l = j+b
- Now weights depend on "where to look relative to output position"

**Key insight for translation invariance:**
- If shifting input should shift output...
- Then V cannot depend on (i,j)
- Same weights must be used everywhere!

**This is exactly what convolution does.**
-->

---

# Mathematical Formulation of Translation Invariance
<div></div>

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}\end{aligned}$$

* We have $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ and $\mathbf{U}$ is a constant, say $u$
	* As a result, we can simplify the definition for $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}$$

* This is a **convolution**!
We are effectively weighting pixels at $(i+a, j+b)$
in the vicinity of location $(i, j)$ with coefficients $[\mathbf{V}]_{a, b}$
to obtain the value $[\mathbf{H}]_{i, j}$

<br>

#### Note that $[\mathbf{V}]_{a, b}$ needs many fewer coefficients than $[\mathsf{V}]_{i, j, a, b}$ since it no longer depends on the location within the image

<!--
**Presenter Notes:**

**The simplification:**
- V[i,j,a,b] becomes just V[a,b]
- We removed the dependency on position (i,j)
- Same weights used at every spatial location

**This IS a convolution!**
- We weight nearby pixels by learned coefficients
- Sum them up to get the output
- Repeat at every position with SAME weights

**Parameter reduction:**
- Before: O(n² × n²) = O(n⁴) where n = image size
- After: O(n²) - just depends on how far we look (a,b range)

Still too many parameters! Next: locality.
-->


---

# Mathematical Formulation of Locality

<div class="grid grid-cols-[5fr_3fr] gap-5">
<div>

* We believe that we should not have to look very far from location $(i, j)$ to understand what's happening at $[\mathbf{H}]_{i, j}$
	* This means outside some range $|a|> \Delta$ or $|b| > \Delta$,
we set $[\mathbf{V}]_{a, b} = 0$
	* The convolution becomes:

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}$$

* This is a **convolutional layer**<br> with kernel size $(2\Delta + 1)$!

</div>
<div>
<center>

#### Parameter reduction:
</center>

| NN/Stage | # of parameters |
|-------|------------|
| Full MLP | $10^{12}$ |
| + Translation invariance | $4 \times 10^6$ |
| + Locality<br> ($\Delta=5$) | $\small(2 \times 5 + 1)^2$<br>$= 121$ |

**~$10^{10}$ reduction!**

</div>
</div>

<!--
**Presenter Notes:**

**Locality principle:**
- To understand what's at position (i,j), we only need nearby pixels
- Set V[a,b] = 0 for |a| > Δ or |b| > Δ
- This defines the **kernel size**: (2Δ+1) × (2Δ+1)

**Parameter counting table:**
- Full MLP: 10^12 (impossible)
- + Translation invariance: 4 × 10^6 (still huge)
- + Locality (Δ=5, so 11×11 kernel): 121 parameters!

**10 billion times fewer parameters!**

This is why CNNs work:
1. Exploit structure (translation invariance)
2. Exploit locality (nearby pixels matter most)
3. Massive parameter reduction → better generalization

**Typical Δ values:** 1 (3×3), 2 (5×5), 3 (7×7)
-->

---
zoom: 0.95
---

# Convolutions
#### Why such equation is called **convolution**?

* In mathematics, the *convolution* between two functions, say $f, g: \mathbb{R}^d \to \mathbb{R}$ is defined as
$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}$$
* We measure the overlap between $f$ and $g$ when one function is "flipped" and shifted by $\mathbf{x}$
* Whenever we have discrete objects, the integral turns into a sum:
$$(f * g)(i) = \sum_a f(a) g(i-a)$$
* In 2D, we have a corresponding sum
with indices $(a, b)$ for $f$ and $(i-a, j-b)$ for $g$:
$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)$$

<!--
**Presenter Notes:**

**Mathematical definition of convolution:**

**Continuous case:**
- Integral over all space
- One function is "flipped" and shifted
- Measures overlap as we shift

**Discrete case:**
- Integral becomes sum
- Same idea: flip, shift, multiply, sum

**Why "flip"?**
- Mathematical convolution has (i-a), not (i+a)
- This ensures commutativity: f*g = g*f
- In DL we use cross-correlation (i+a) but call it convolution
- Doesn't matter since we learn the kernel!

**Key applications of convolution:**
- Signal processing (filtering)
- Image processing (blur, sharpen, edge detect)
- Probability (sum of random variables)
- Deep learning (feature extraction)
-->

