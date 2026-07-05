# Parametric Curve Parameter Estimation

## Problem

Find the unknowns $\theta$, $M$, $X$ in:

$$x(t) = t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X$$

$$y(t) = 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)$$

given:
- $0° < \theta < 50°$, $-0.05 < M < 0.05$, $0 < X < 100$
- parameter range $6 < t < 60$
- `xy_data.csv`: 1500 $(x, y)$ points that lie on the curve, **in no particular order** (not sorted by $t$)

## Final answer

| Variable | Value |
|---|---|
| $\theta$ | **30°** (0.5236 rad) |
| $M$ | **0.03** |
| $X$ | **55** |

Mean L1 distance from every data point to the nearest point on the fitted curve: **0.0069** (max 0.021) — this residual is just numerical grid-discretization noise, i.e. an essentially exact fit.

**Desmos expression** (paste into desmos.com/calculator, set $t$ domain to $[6, 60]$):

```
(t*cos(0.5236)-e^{0.03|t|}*sin(0.3t)*sin(0.5236)+55, 42+t*sin(0.5236)+e^{0.03|t|}*sin(0.3t)*cos(0.5236))
```

## Approach — step by step

### 1. Spot the hidden structure

Rewriting the two equations:

$$x - X = t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta)$$

$$y - 42 = t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)$$

Let $u = t$ and $v = e^{M|t|}\sin(0.3t)$. Then:

$$\begin{pmatrix} x - X \\ y - 42 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}$$

So the data cloud is just a **2D rotation by $\theta$ + translation $(X, 42)$** applied to the simple base curve $(u, v) = (t,\ e^{Mt}\sin(0.3t))$ (for $t>0$, $|t|=t$). Since $t$ ranges up to 60 while $v$ is bounded (roughly $\pm e^{0.05 \times 60} \approx \pm 20$ at the extreme of the $M$ range), the base curve is close to a straight diagonal line with a wiggle riding on top — the wiggle's amplitude growth rate is controlled by $M$.

This matters practically because it tells us the point cloud, once "un-rotated," should collapse onto a near-linear shape — a useful sanity check.

### 2. Why not plain least-squares?

Normally, if you had $(t_i, x_i, y_i)$ triples, you could fit $\theta, M, X$ directly with least squares. But the CSV gives only $(x, y)$ — **no $t$ column**, and the rows are shuffled (checked: consecutive rows are far apart in distance, so they are not in $t$-order). So there's no point-to-point correspondence to regress against.

### 3. PCA sanity check

Running PCA on the raw $(x,y)$ cloud, the dominant principal axis sits at $\approx 28.5°$ (mod 180°) — consistent with the rotation angle living inside the given $[0°, 50°]$ range. This isn't the final answer (it ignores the wiggle and the exact metric used for grading), but it's a fast confidence check before running a heavier optimizer.

### 4. Global optimization with a Chamfer-style loss

Since there's no index correspondence, the fitting objective instead measures, for a candidate $(\theta, M, X)$:

1. Densely sample the candidate curve over $t \in [6, 60]$.
2. Build a KD-tree over those sampled curve points.
3. For every one of the 1500 data points, find the distance to its *nearest* point on the candidate curve.
4. Loss = mean of those distances (L1 norm) — mirroring the grading metric described in the assessment.

This loss is minimized with:
- `scipy.optimize.differential_evolution` — a global, bounded, derivative-free optimizer. Chosen because the $\sin(0.3t)$ term makes the loss landscape non-convex/wiggly, so a purely local method risks getting stuck.
- Followed by `Nelder-Mead` local polishing for extra precision.

### 5. Result & verification

The optimizer converges cleanly to $\theta=30°$, $M=0.03$, $X=55$ — clean round numbers with near-zero residual, which is itself a strong signal these are the exact intended values rather than an approximate local optimum. Plotting the fitted curve against the raw data confirms a visually perfect overlay:

![Fit verification](assets/fit_check.png)

## Files

- `fit_curve.py` — full pipeline (PCA check, optimization, verification plot)
- `xy_data.csv` — provided dataset (add this file alongside the script)
- `requirements.txt` — dependencies
- `assets/fit_check.png` — fitted curve vs. data overlay

## Reproduce

```bash
pip install -r requirements.txt
python fit_curve.py
```
