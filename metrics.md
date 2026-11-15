# Evaluation metrics for the two-phase flow benchmark

This note defines the scores used to evaluate a surrogate model: two accuracy scores (pressure and saturation) and two physical-consistency scores (mass conservation and saturation bounds). Every score is a ratio of integrals over space and time. It is computed for each test sample and then averaged over the test samples.

## 1. Notation

- $\Omega = \Omega^2 \cup \Omega^1 \cup \Omega^0$ is the mixed-dimensional domain: rock matrix (2D), fractures (1D) and fracture intersections (0D).
- $j = 1, \dots, N$ are the test samples, each a pair $(K_{45}, h)$.
- $t_k$ are the times at which the fields are stored. They are equally spaced, with step $\Delta t$.
- $u$ is a reference (full-order) field and $\hat u$ its prediction, with $u \in \lbrace p, S_0 \rbrace$.
- $u_i^k$ is the value of $u$ in cell $i$ at time $t_k$. The index $i$ runs over the cells of all subdomains.

Unless stated otherwise, every quantity refers to one test sample.

## 2. Integrals and norms

**Space.** The integral over $\Omega$ adds the contributions of all subdomains, each weighted by its specific volume $\nu$: 1 in the matrix, the aperture $\varepsilon$ in a fracture and $\varepsilon^2$ at an intersection, with $\varepsilon = 10^{-2}$.

$$
\int_\Omega f \mathrm{d}\Omega = \sum_{d=0}^{2} \int_{\Omega^d} \nu f \approx \sum_i w_i f_i, \qquad w_i = \nu_i \lvert K_i \rvert
$$

Here $\lvert K_i \rvert$ is the measure of cell $i$ (area, length, or 1 for a point), so $w_i$ is the volume of the cell per unit thickness. The weights change from one test sample to another, because the fault throw $h$ changes the geometry. The total volume is $\lvert \Omega \rvert = \sum_i w_i$: the matrix holds 98% of it, the fractures 2% and the intersections 0.03%.

**Time.** The snapshots are equally spaced, so the integral over a time window $I$ is the sum over its snapshots times the step:

$$
\int_I g \mathrm{d}t \approx \Delta t \sum_{k \in I} g(t_k)
$$

$\Delta t$ cancels in every score below, so the discrete formulas are written without it.

**Norms.** For a field $f$, in space and in space-time:

$$
\lVert f \rVert_{\Omega}^2 = \int_\Omega f^2 \mathrm{d}\Omega \approx \sum_i w_i f_i^2, \qquad \lVert f \rVert_{I \times \Omega}^2 = \int_I \lVert f \rVert_{\Omega}^2 \mathrm{d}t \approx \Delta t \sum_{k \in I} \sum_i w_i (f_i^k)^2
$$

For a function of time $g$:

$$
\lVert g \rVert_{I}^2 = \int_I g^2 \mathrm{d}t \approx \Delta t \sum_{k \in I} g(t_k)^2
$$

**Time windows.** The accuracy scores are computed on two windows: $I_\mathrm{in}$, the test times inside the training time range, and $I_\mathrm{ex}$, the test times after the last training time (extrapolation). The physical scores are computed once, on the whole test time range.

**Test samples.** A score $E$ is computed for each test sample and then averaged, $\bar E = \frac{1}{N} \sum_j E^{(j)}$. Since the samples are drawn uniformly, this approximates the mean of $E$ over the parameter space.

## 3. Accuracy

For $u = p$ and for $u = S_0$, the score is the relative $L^2$ error in space and time:

$$
E_u = \frac{\lVert \hat u - u \rVert_{I \times \Omega}}{\lVert u \rVert_{I \times \Omega}} \approx \sqrt{\frac{\sum_{k \in I} \sum_i w_i (\hat u_i^k - u_i^k)^2}{\sum_{k \in I} \sum_i w_i (u_i^k)^2}}
$$

The error is normalized once per trajectory by a space-time quantity, not snapshot by snapshot or subdomain by subdomain. This denominator cannot vanish, so no safeguard against a division by zero is needed.

For pressure, the field is positive everywhere.

For saturation, the Cauchy-Schwarz inequality gives, at every time,

$$
\lVert S_0 \rVert_{\Omega} \ge \frac{1}{\lvert \Omega \rvert^{1/2}} \int_\Omega S_0 \mathrm{d}\Omega
$$

The integral on the right is proportional to the mass of phase 0, which is positive and constant in time (Section 4.1).

## 4. Physical consistency

### 4.1 Mass conservation

The mass of phase 0 in the domain is

$$
M(t) = \int_\Omega \phi \rho_0 S_0 \mathrm{d}\Omega
$$

The porosity $\phi$ and the density $\rho_0$ are constant, so they cancel in the score and only the integral of the saturation is needed:

$$
m(t) = \int_\Omega S_0 \mathrm{d}\Omega \approx \sum_i w_i S_{0,i}^k
$$

$\hat m$ is the same quantity computed with $\hat S_0$. The score is the relative error of the mass over the whole test time range:

$$
E_M = \frac{\lVert \hat m - m \rVert_{I}}{\lVert m \rVert_{I}} \approx \sqrt{\frac{\sum_{k \in I} (\hat m^k - m^k)^2}{\sum_{k \in I} (m^k)^2}}
$$

- The normalization is the mass itself, which never vanishes. The domain is closed and the full-order scheme conserves mass, so $m$ is constant in time and $E_M$ is the root-mean-square drift of the predicted mass relative to the true mass.
- $E_M$ measures the net amount of phase 0 that the surrogate creates or destroys. It does not see fluid that is in the wrong place, which is what $E_{S_0}$ measures.
- Phase 1 needs no separate score: since $S_1 = 1 - S_0$, an excess volume of phase 0 is an equal deficit of phase 1.

### 4.2 Saturation bounds

The out-of-bounds part of the predicted saturation is its distance from the interval $[0, 1]$:

$$
v = \max(-\hat S_0, \hat S_0 - 1, 0)
$$

The score uses the same norm and the same normalization as the saturation error, on the whole test time range:

$$
E_B = \frac{\lVert v \rVert_{I \times \Omega}}{\lVert S_0 \rVert_{I \times \Omega}}
$$

- $E_B = 0$ if and only if $0 \le \hat S_0 \le 1$ in every cell and at every time.
- The reference satisfies the bounds, so $v \le \lvert \hat S_0 - S_0 \rvert$ in every cell. $E_B$ therefore cannot exceed the saturation error computed on the same time range: it is the part of that error that lies outside the physical range.

## 5. Reported scores

| Score | Measures | Time range | Defined in |
|---|---|---|---|
| $\bar E_p$ | pressure error | $I_\mathrm{in}$ and $I_\mathrm{ex}$ | Section 3 |
| $\bar E_{S_0}$ | saturation error | $I_\mathrm{in}$ and $I_\mathrm{ex}$ | Section 3 |
| $\bar E_M$ | mass-conservation error | whole test range | Section 4.1 |
| $\bar E_B$ | violation of the saturation bounds | whole test range | Section 4.2 |

Six numbers in total, all dimensionless and all equal to zero for a perfect surrogate.

## 6. Properties of the dataset used above

Checked on the full-order data and model on 2026-11-15.

- The snapshots are stored every $\Delta t = 0.02$: 68 per training sample (up to $t = 1.34$) and 100 per test sample (up to $t = 1.98$).
- The time is measured from the first stored snapshot, which is given as an input. The simulation starts 0.02 earlier, from an initial condition that is not in the data.
- The boundaries are closed and there are no source terms. In the 20 test samples the mass of phase 0 changes by less than $1.3 \cdot 10^{-6}$ of its value over the whole simulation.
- The porosity is 0.25 everywhere, and the densities are constant (compressibility $10^{-10}$).
- The aperture is $10^{-2}$ in all fractures and intersections.
- The reference saturation lies in $[0, 1]$ and the reference pressure in $[0.96, 2.07]$.
- The `cell_volumes` of the HDF5 files are the weights $w_i$. The original simulation files hold $\lvert K_i \rvert$ only (area, length, and 1 for each intersection), and `src/data/extract_data.py` multiplies them by the specific volume.

## 7. Reference values

A surrogate that returns the first state, which is given, for all times predicts no dynamics at all. On the 20 test samples it obtains the following scores.

| Score | $I_\mathrm{in}$ | $I_\mathrm{ex}$ | Whole test range |
|---|---|---|---|
| $\bar E_p$ | 0.95% | 1.7% | |
| $\bar E_{S_0}$ | 27% | 48% | |
| $\bar E_M$ | | | 0 |
| $\bar E_B$ | | | 0 |

- The pressure changes little in time, so its reference is small. A pressure error is informative only when it is well below these values.
- The two physical scores are zero for this surrogate: the first state has the right mass and respects the bounds. They complement the accuracy scores and cannot replace them.
