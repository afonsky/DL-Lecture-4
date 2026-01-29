---
layout: center
---

<center>

# Extra material:<br> Loss Functions for Semantic Segmentation
</center>

---

# What is Semantic Segmentation?

<div>
  <figure>
    <img src="/semantic_segmentation.gif" style="width: 400px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Illustration of semantic segmentation.<br> Image source:
      <a href="https://hackmd.io/@gianghoangcotai/ryCqF_uO8">https://hackmd.io/@gianghoangcotai/ryCqF_uO8</a>
    </figcaption>
  </figure>   
</div>
<br>
<br>

The aim of semantic segmentation: to classify each pixel of the input image (or video) into one of the predefined classes. In other words, given a colored image $\bm{I}_C \in \bm{R}^{M \times N \times C}$, we want to produce a probability mask $\bm{P} \in \bm{R}^{H \times W \times N_C}$ where $N_C$ is the number of predefined classes, $H \times W$ - height and width of the image.

---

# Loss Functions for Semantic Segmentation
<div></div>

Cross-entropy loss has been standard loss function for semantic segmentation, as well as image classification. Recall that the cross-entropy loss between a probability vector $p \in \bm{R}^{N_C}$ and a onehot vector $\bm{y} \in \{0,1\}^{N_C}$ is defined as:

$$L(\bm{p}, \bm{y}) = -\sum\limits_{i=0}^{N_C-1} \bm{y}_i \cdot \mathrm{log} \bm{p}_i$$


Now suppose that our model outputs a probability mask $\bm{P} \in \bm{R}^{H \times W \times N_C}$, and the onehot ground-truth segmentation mask is $\bm{Y} \in \{0,1\}^{H \times W \times N_C}$ then our loss function would be:

$$L(\bm{P}, \bm{Y}) = -\sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1} \sum\limits_{k=0}^{N_C-1} \bm{Y}_{ijk} \cdot \mathrm{log} \bm{P}_{ijk}$$

<span style="color:grey"><small> Based on the [CoTAI lecture](https://hackmd.io/@gianghoangcotai/ryCqF_uO8)</small></span>

---

# Loss Functions for Semantic Segmentation

### Weighted Cross Entropy Loss

To reduce the domination of background pixels in the loss function, we use weighted cross entropy loss: $L(\bm{P}, \bm{Y}) = -\sum\limits_{i=0}^{H-1} \sum\limits_{j=0}^{W-1} \sum\limits_{k=0}^{N_C-1} w_k \cdot \bm{Y}_{ijk} \cdot \mathrm{log} \bm{P}_{ijk}$

### Focal Loss

<small>Focal loss is another loss function designed to address the problem of class imbalance. It originates from object detection problem, where the number of negative bounding boxes, i.e. boxes containing no objects, domiates the cross-entropy loss function.</small>

Let $p$ be a probability of positive class $1$, and $y \in \{0, 1\}$ be the ground-truth class. The focal loss for binary classification is defined as:

$L = -(1 - p)^\alpha \cdot y \cdot \mathrm{log} p - p^\alpha \cdot (1 - y) \cdot \mathrm{log} (1 - p)$, where $\alpha$ is a hyperparameter. 

<span style="color:grey"><small> Based on the [CoTAI lecture](https://hackmd.io/@gianghoangcotai/ryCqF_uO8)</small></span>

---

# Loss Functions for Semantic Segmentation
<div></div>

## Localization losses

* DICE Loss: $L(\bm{P}, \bm{Y}) = \sum\limits_{c=0}^{N_C-1} L_C(\bm{P}, \bm{Y})$, where $L_c(\mathbf{P}, \mathbf{Y})=-\log\bigg(\frac{2\cdot |\mathbf{P}_{:,:,c} \circ \mathbf{Y}_{:,:,c}|}{|\mathbf{P}_{:,:,c}|+|\mathbf{Y}_{:,:,c}|}\bigg)$

* Intersection over Union (IoU) Loss: $L(\bm{P}, \bm{Y}) = \sum\limits_{c=0}^{N_C-1} L_C(\bm{P}, \bm{Y})$, <br>where $L_c(\mathbf{P}, \mathbf{Y})=-\log\bigg(\frac{|\mathbf{P}_{:,:,c} \circ \mathbf{Y}_{:,:,c}|}{|\mathbf{P}_{:,:,c}|+|\mathbf{Y}_{:,:,c}|-|\mathbf{P}_{:,:,c} \circ \mathbf{Y}_{:,:,c}|}\bigg)$

<div class="grid grid-cols-[1fr_1fr]">
<div>
<br>
<br>
<br>

<span style="color:grey"><small> Based on the [CoTAI lecture](https://hackmd.io/@gianghoangcotai/ryCqF_uO8)</small></span>

</div>
<div>
  <figure>
    <img src="/iou_equation.png" style="width: 220px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Illustration of IoU score. Image by
      <a href="https://people.cs.pitt.edu/~kovashka">Adriana Kovashka</a>
    </figcaption>
  </figure>   
</div>
</div>

---

# Loss Functions for Semantic Segmentation

<div class="grid grid-cols-[2fr_1fr]">
<div>

#### Boundary Loss. [Hausdorff Distance](https://en.wikipedia.org/wiki/Hausdorff_distance):
* Distance map $\bm{D} \in \bm{R}^{H \times W}$,<br> where $\bm{D}_{ij}$ indicates the distance<br> from pixel $(i,j)$ to its nearest boundary point.
* Given the distance map $\bm{D}$, we normalize it to $[0,1]$ by $\forall(i,j),\mathbf{D}_{ij}=\frac{\mathbf{D}_{ij} - \min_{st} \mathbf{D}_{st}}{\max_{st} \mathbf{D}_{st} - \min_{st} \mathbf{D}_{st}}$
</div>
<div>
  <figure>
    <img src="/boundary_pixels.png" style="width: 220px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Pixels along object boundaries are marked white, i.e. ignored.<br> Image from
      <a href="https://hackmd.io/@gianghoangcotai/ryCqF_uO8">CoTAI lecture</a>
    </figcaption>
  </figure>   
</div>
</div>
<br>

<div class="grid grid-cols-[1fr_2fr]">
<div>
  <figure>
    <img src="/water_dt.jpg" style="width: 220px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute;"><br>Distance map (left) from every pixel to its corresponding nearest object boundary point. Image from
      <a href="https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html">OpenCV tutorial</a>
    </figcaption>
  </figure>   
</div>
<div>

### Hausdorff Distance Loss:
$L(\mathbf{P},\mathbf{Y}) = \allowbreak \\ \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \sum_{k=0}^{N_C-1} (1 - \mathbf{D}_{ij})^\alpha\cdot(\mathbf{P}_{ijk}-\mathbf{Y}_{ijk})^2$

</div>
</div>