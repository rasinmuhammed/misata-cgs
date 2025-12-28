# Reverse Graph Breakthrough: Pixels → Probability Distributions

## Vision
Reconstruct generative models from visual artifacts (charts, histograms, PDFs) when raw source data is inaccessible due to privacy/legal/technical silos.

---

## Visual Extraction Pipeline

### State-of-the-Art: DePlot & MatCha

| Model | Function | Output |
|-------|----------|--------|
| **DePlot** | Modality translation | Image → Structured table/Markdown |
| **MatCha** | Mathematical reasoning | Handles implicit/occluded data points |
| **LineFormer** | Vector extraction | Transformer-based line coordinate prediction |
| **ChartDete** | Object detection | Axes/keypoints identification |

---

## Algorithm: PDF Reconstruction & Sampling

```
┌─────────────────┐
│  Chart Image    │
└────────┬────────┘
         ▼
┌─────────────────────────────────┐
│ 1. VISUAL PARSING (DePlot)     │
│    Extract: D = {(x₁,y₁)...}   │
│    + axis labels for scale     │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ 2. CONTINUOUS APPROXIMATION    │
│    Fit: Cubic Spline or KDE    │
│    → g(x) unnormalized PDF     │
│    C = ∫g(x)dx (normalization) │
│    f(x) = g(x)/C               │
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ 3. CDF CONSTRUCTION            │
│    F(x) = ∫f(t)dt from -∞ to x │
│    F(x) is monotonic → invertible│
└────────┬────────────────────────┘
         ▼
┌─────────────────────────────────┐
│ 4. INVERSE TRANSFORM SAMPLING  │
│    u ~ Uniform(0,1)            │
│    x_gen = F⁻¹(u)              │
│    → Sample from visual dist!  │
└─────────────────────────────────┘
```

---

## Key Insight
> **"Hydrate dead pixels into an infinite stream of statistically accurate synthetic data"**
>
> Clone system behavior from a screenshot of its performance metrics.

---

## References
- DePlot: [arXiv 2212.10505](https://arxiv.org/abs/2212.10505)
- MatCha: [Google Research Blog](https://research.google/blog/foundation-models-for-reasoning-on-charts/)
- Inverse Transform Sampling: [Brilliant Wiki](https://brilliant.org/wiki/inverse-transform-sampling/)
