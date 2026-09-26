---
theme: seriph
addons:
  - "@twitwi/slidev-addon-ultracharger"
addonsConfig:
  ultracharger:
    inlineSvg:
      markersWorkaround: false
    disable:
      - metaFooter
      - tocFooter
background: /logo/mountain.jpg
highlighter: shiki
routerMode: hash
lineNumbers: false
duration: 80min

css: unocss
title: Deep Learning
subtitle: Convolutional Neural Networks
date: 28/09/2026
venue: HSE
author: Alexey Boldyrev
---

# <span style="font-size:28.0pt" v-html="$slidev.configs.title?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:32.0pt" v-html="$slidev.configs.subtitle?.replaceAll(' ', '<br/>')"></span>
# <span style="font-size:18.0pt" v-html="$slidev.configs.author?.replaceAll(' ', '<br/>')"></span>

<span style="font-size:18.0pt" v-html="$slidev.configs.date?.replaceAll(' ', '<br/>')"></span>

<div class="abs-tl mx-5 my-10">
  <img src="/logo/FCS_logo_full_L.svg" class="h-18">
</div>

<div class="abs-tr mx-5 my-5">
  <img src="/logo/DSBA_logo.png" class="h-28">
</div>

<style>
  :deep(footer) { padding-bottom: 3em !important; }
</style>


---
src: ./slides/0_introduction.md
---

---
src: ./slides/1_convolutional_layer.md
---

---
src: ./slides/2_padding_stride_channels.md
---

---
src: ./slides/3_pooling_layer.md
---

---
src: ./slides/4_CNNs.md
---

---
src: ./slides/5_conclusions.md
---

---
src: ./slides/6_semantic_segmentation.md
---

---
src: ./slides/0_backup.md
---

---
src: ./slides/0_backup_math.md
---

---
src: ./slides/0_backup_lecture5.md
---

---
src: ./slides/0_end.md
---