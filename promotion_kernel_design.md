# Adding a promotion state to the kernel interpolation

## 1. Current formulation

The controller interpolates a **prescribed acceleration** over a feature point

$$\varphi = (g_i,\ \tau_i,\ \gamma_i), \qquad \gamma_i \in \{0,1\}\ \ (0=\text{yielder},\ 1=\text{passer}).$$

- **Anchors** $X=\{x_m\}_{m=1}^{M}$ are the grid
  $\texttt{\_G\_LEVELS}\times\texttt{\_TAUC\_LEVELS}\times\texttt{\_ROLE\_LEVELS}$, so
  $M = 12\times 4\times 2 = 96$ points (`ANCHOR_FEATS`).
- Each anchor carries a **prescribed target**
  $y_m = \texttt{\_anchor\_target}(g,\tau,\gamma) \in \{\text{brake},\,0,\,\text{free},\,\text{clear},\,\text{yield}\}.$
- ARD Gaussian kernel between two feature points:

$$k(\varphi,\varphi') = \exp\!\left(-\sum_{d} \frac{(\varphi_d-\varphi'_d)^2}{2\,\ell_d^2}\right),
\qquad \ell = (\ell_g,\ \ell_\tau,\ \ell_\gamma)\ \ (\texttt{LENGTHSCALES}).$$

- Gram matrix $K_{mn}=k(x_m,x_n)$, and the output (with optional learned prior mean $f$):

$$a(\varphi_*) = f(\varphi_*) + k(\varphi_*)^\top K^{-1}\big(y - f(X)\big).$$

At an anchor $\varphi_*=x_m$ the correction cancels $f$ exactly and returns $y_m$, so the anchors pin the physics targets per-context.

---

## 2. Proposed extension: a 4th coordinate $p$

Add a **promotion state** $p_i \in \{0,1\}$ (0 = normal, 1 = promoted):

$$\boxed{\ \varphi = (g_i,\ \tau_i,\ \gamma_i,\ p_i)\ }$$

### 2.1 Anchors double

$$X' = X \times \{0,1\}, \qquad M' = 2M = 192.$$

Every existing anchor is duplicated at $p=0$ and $p=1$.

### 2.2 Kernel gains a 4th lengthscale

$$k(\varphi,\varphi') = \exp\!\left(-\frac{(g-g')^2}{2\ell_g^2}-\frac{(\tau-\tau')^2}{2\ell_\tau^2}-\frac{(\gamma-\gamma')^2}{2\ell_\gamma^2}-\frac{(p-p')^2}{2\ell_p^2}\right).$$

Choosing $\ell_p$ **small** decouples the two sheets: the factor $\exp(-1/2\ell_p^2)\to 0$, so a query at $p=0$ interpolates essentially only the $p=0$ anchors and a query at $p=1$ only the $p=1$ anchors. Promotion becomes a clean switch between two prescribed fields, entirely inside the kernel.

### 2.3 Targets

$$y'_m = \texttt{\_anchor\_target}(g,\tau,\gamma,p) =
\begin{cases}
\texttt{\_anchor\_target}(g,\tau,\gamma) & p = 0 \quad(\text{unchanged, today's behavior})\\[4pt]
\text{promotion targets below} & p = 1
\end{cases}$$

Proposed $p=1$ targets — promoted vehicle **passes** but stays physically safe:

$$\texttt{\_anchor\_target}(g,\tau,\gamma,\,p{=}1)=
\begin{cases}
\textbf{brake} & g < 1 \quad(\text{keep longitudinal / rear-end safety})\\[2pt]
\textbf{clear}=a_{\max} & \tau = 0 \quad(\text{assert through the conflict point})\\[2pt]
\textbf{free} & g \ge 1 \quad(\text{free-flow, ignore the cross timing }\tau)
\end{cases}$$

In words: **promotion removes the $\tau$-gated cross yield** (the vehicle no longer brakes for crossing traffic) while keeping car-following and the $g<1$ brake. So a promoted vehicle accelerates through the box; it only cannot rear-end a leader.

---

## 3. Why this is the right place for it

- The promotion is **prescribed**, not patched into the SUMO loop. The same kernel evaluated with $p_i=1$ yields a "go" command; with $p_i=0$ the normal negotiation. No special-case gate.
- It is **per-vehicle**: each vehicle's own $p_i$ flips its prescribed field, so a promoted compatible group all get "free/clear" at once while others keep $p=0$.
- It composes with the existing GP correction / learned mean exactly as before (anchors still pin targets).

---

## 4. Step-by-step implementation plan (`utils.py` only)

1. `_PROMO_LEVELS = (0.0, 1.0)`; make `ANCHOR_FEATS` 4-tuples $(g,\tau,\gamma,p)$.
2. Add the $p=1$ branch to `_anchor_target`.
3. Append $\ell_p$ to `LENGTHSCALES`; rebuild $K^{-1}$ over the 4D anchors (cache is shape-keyed, so it refreshes).
4. `controller_acceleration` stacks $p_i$ into $\varphi$ (the per-vehicle promotion flag) and sweeps the 4D anchor counterfactuals.
5. (Later) decide whether the learned mean $f$ sees $p$ or stays 3-D.

---

## 5. Two things to confirm before coding

1. **The $p=1$ targets** in §2.3 — is "free/clear on the cross dimension, keep the $g<1$ brake" the promotion behavior you want?
2. **Does the learned mean $f$ see $p$**, or stay 3-D (promotion lives only in the prescribed anchors)?
