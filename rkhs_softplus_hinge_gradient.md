# RKHS gradient of $\operatorname{softplus}(\tau_g)^2$ w.r.t. $a_i$

## Setup

The gap functional is

$$
\tau_g(i,t) = \tau_{\text{safety}}
- \frac{d_{ij}(t)}{\,v_i(t)+\int_{t}^{\tau_i} a_i(s)\,ds\,}
- \frac{d_{i'j}(t)}{\,v_{i'}(t)+\int_{t}^{\tau_{i'}} a_{i'}(s)\,ds\,}
- t_{\text{safety}} .
$$

Define the predicted (integrated) speed of car $i$ at the conflict point

$$
V_i(t) \;=\; v_i(t) + \int_{t}^{\tau_i} a_i(s)\,ds ,
$$

so that the only term that depends on $a_i$ is $-\,d_{ij}(t)/V_i(t)$. The
$i'$ term depends on $a_{i'}$, **not** $a_i$, so it drops out of the gradient.

The loss is

$$
L \;=\; \operatorname{softplus}\big(\tau_g(i,t)\big)^2,
\qquad
\operatorname{softplus}(x)=\log(1+e^{x}).
$$

## Scalar (chain-rule) factor

Let $\phi(x)=\operatorname{softplus}(x)^2$. Then

$$
\phi'(x) \;=\; 2\,\operatorname{softplus}(x)\,\sigma(x),
\qquad
\sigma(x)=\operatorname{softplus}'(x)=\frac{1}{1+e^{-x}} .
$$

Write the scalar coefficient

$$
\boxed{\;
c \;=\; \phi'(\tau_g)\,\frac{d_{ij}(t)}{V_i(t)^2}
\;=\; 2\,\operatorname{softplus}(\tau_g)\,\sigma(\tau_g)\,\frac{d_{ij}(t)}{V_i(t)^2}
\;}
$$

because

$$
\frac{\partial \tau_g}{\partial V_i} = \frac{d_{ij}}{V_i^2},
\qquad
\frac{\partial L}{\partial V_i} = \phi'(\tau_g)\,\frac{d_{ij}}{V_i^2} = c .
$$

Note $c \ge 0$ always (softplus, $\sigma$, $d_{ij}$, $V_i^2$ are all nonnegative),
so the correction only ever *raises* $V_i$, i.e. brakes/accelerates to open the gap.

## Plain (L²) functional gradient

Since $\displaystyle \frac{\partial V_i}{\partial a_i(s)} = \mathbb{1}[\,t \le s \le \tau_i\,]$,

$$
\frac{\partial L}{\partial a_i(s)}
\;=\; c \,\cdot\, \mathbb{1}[\,t \le s \le \tau_i\,].
$$

This is just a constant pulse over the prediction horizon $[t,\tau_i]$.

## RKHS gradient

In an RKHS $\mathcal H$ with reproducing kernel $k(\cdot,\cdot)$, the reproducing
property $a_i(s)=\langle a_i,\,k(s,\cdot)\rangle_{\mathcal H}$ turns the integral
functional into an inner product:

$$
\int_{t}^{\tau_i} a_i(s)\,ds
= \Big\langle a_i,\; \psi \Big\rangle_{\mathcal H},
\qquad
\psi(\cdot) \;=\; \int_{t}^{\tau_i} k(s,\cdot)\,ds .
$$

So $\nabla_{\mathcal H} V_i = \psi$, and by the chain rule the **RKHS gradient** is

$$
\boxed{\;
\nabla_{\mathcal H} L \,(\cdot)
\;=\; c\;\psi(\cdot)
\;=\; 2\,\operatorname{softplus}(\tau_g)\,\sigma(\tau_g)\,
\frac{d_{ij}(t)}{V_i(t)^2}
\int_{t}^{\tau_i} k(s,\cdot)\,ds
\;}
$$

evaluated at a point $s'$:

$$
\nabla_{\mathcal H} L \,(s')
= 2\,\operatorname{softplus}(\tau_g)\,\sigma(\tau_g)\,
\frac{d_{ij}(t)}{V_i(t)^2}
\int_{t}^{\tau_i} k(s,s')\,ds .
$$

## Reading it

- **Same scalar, different shape.** The RKHS gradient is the *same* scalar $c$
  as the L² case, but the indicator pulse $\mathbb{1}[t\le s\le\tau_i]$ is
  replaced by its **kernel-smoothed** version $\psi(s')=\int_t^{\tau_i}k(s,s')\,ds$.
  Equivalently, $\nabla_{\mathcal H}L = K\big(\partial L/\partial a_i\big)$:
  the RKHS gradient is the kernel (integral) operator applied to the plain L² gradient.

- **Effect.** Instead of a hard box over $[t,\tau_i]$, you get a smooth
  acceleration correction whose temporal spread is set by the kernel bandwidth —
  a smoother, traceable control update.

- **RBF example.** With $k(s,s')=\exp\!\big(-(s-s')^2/2\ell^2\big)$,

$$
\psi(s') = \int_{t}^{\tau_i} e^{-(s-s')^2/2\ell^2}\,ds
= \ell\sqrt{\tfrac{\pi}{2}}
\left[\operatorname{erf}\!\Big(\tfrac{\tau_i-s'}{\sqrt2\,\ell}\Big)
-\operatorname{erf}\!\Big(\tfrac{t-s'}{\sqrt2\,\ell}\Big)\right].
$$
