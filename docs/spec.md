# Thermodynamic Susceptibilities for Learning Systems

## One-Sentence Summary

This is a finite MNIST pilot for asking where validation risk is most sensitive
to data, capacity, and regularization controls. The useful output is a response
map, not a claim that the network already has a literal thermodynamic free
energy.

## Aim

A heatmap of validation loss says which trained models performed best. The
object of interest here is the local response surface around that heatmap:
where does the model still benefit from more data, more width, longer training,
or stronger regularization?

The thermodynamic analogy is deliberately modest. In an ordinary physical phase
diagram, a specific heat peak marks a regime where a control variable suddenly
becomes more or less effective. Here we use the same shape of question for a
learning system: where do finite perturbations of training controls produce
large changes in validation risk?

For a fixed training protocol with controls

$$
\theta = (\theta_N,\theta_P,\theta_T,\theta_\lambda,\theta_\eta,\ldots),
$$

we measure a risk-like potential

$$
R(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_{\rm val}}
\left[\ell(f_{\theta}(x), y)\right].
$$

For the MNIST neural-network sweep, $R$ is validation cross-entropy. A local
conjugate-like response is

$$
h_i(\theta) = -\frac{\partial R}{\partial \theta_i},
$$

and the susceptibility or specific-heat-like curve is

$$
C_i(\theta) =
\left|\frac{\partial h_i}{\partial \theta_i}\right|
=
\left|\frac{\partial^2 R}{\partial \theta_i^2}\right|.
$$

Large $h_i$ means that moving along control $i$ still buys down risk. Large
$C_i$ means that the usefulness of that control is changing rapidly. That is the
candidate regime boundary.

## Normalized Coordinates

The controls must be normalized before comparing responses. "Add one datapoint"
and "add one hidden unit" are not comparable perturbations, so the report uses
fractional or logarithmic coordinates:

$$
\theta_N = \log_2 N,\qquad
\theta_P = \log_2 P,\qquad
\theta_\lambda = \log_{10}\lambda .
$$

The plotted quantities therefore have direct finite-perturbation meanings:

$$
h_N = -\frac{\partial R}{\partial \log_2 N}
\quad\text{risk decrease per data doubling},
$$

$$
h_P = -\frac{\partial R}{\partial \log_2 P}
\quad\text{risk decrease per parameter doubling},
$$

$$
h_\lambda = -\frac{\partial R}{\partial \log_{10}\lambda}
\quad\text{risk decrease per decade stronger weight decay}.
$$

For regularization, the sign matters. If $h_\lambda>0$, stronger weight decay
helps locally and the cell is under-regularized. If $h_\lambda<0$, stronger
weight decay hurts locally and the cell is over-regularized. A peak in

$$
C_\lambda =
\left|\frac{\partial h_\lambda}{\partial \log_{10}\lambda}\right|
$$

would be the regularization analog of a specific heat peak.

## Protocol

Different fixed quantities define different ensembles. This report uses a
fixed-epoch local sweep: the number of examples seen per training example is
held fixed, while data size, width, and weight decay vary.

The dense local MNIST run uses:

- scratch retrains;
- a fixed 3-epoch exposure;
- one-hidden-layer ReLU MLPs on cached MNIST, downsampled to $14\times14$;
- fixed optimizer family and batch size;
- a clustered log grid over $N$, $P$, and $\lambda$;
- three independent scratch seeds per condition, with the sampled data subset
  held fixed.

The primary controlled variables are

$$
N\in[8192,50000],\qquad
W\in[32,256],\qquad
\lambda\in[10^{-6},3\times 10^{-2}],
$$

with $\lambda=0$ included as a reference cell but excluded from
$\log_{10}\lambda$ derivatives.

This is still not a thermodynamic limit. It is a local derivative estimator
designed to decide what a larger remote sweep should measure.

## Interpretation

Before seeing the data, the natural labels were:

- A data-limited region has large positive $h_N$ and modest $h_P$.
- A capacity-limited region has large positive $h_P$, especially at larger $N$.
- An under-regularized region has $h_\lambda>0$: increasing weight decay
  improves validation loss.
- An over-regularized region has $h_\lambda<0$: increasing weight decay worsens
  validation loss.
- A clean regularization boundary would appear as a ridge in $C_\lambda$.

In the current MNIST run, the last story is not the clean signal. The median
absolute response scales satisfy roughly

$$
|h_N| \gg |h_\lambda|,
\qquad
|h_P| \gg |h_\lambda|,
$$

and the direct $\lambda$ finite differences are often comparable to seed noise.
The more robust object in this regime is therefore the lower envelope

$$
R^*(N,W)=\min_{\lambda} R(N,W,\lambda),
$$

followed by susceptibilities

$$
h_N^*=-\frac{\partial R^*}{\partial\log_2 N},
\qquad
h_P^*=-\frac{\partial R^*}{\partial\log_2 P}.
$$

This treats regularization as a nuisance knob, asks what still buys down risk
after tuning it, and avoids amplifying a weak $\lambda$ signal into a fake
specific-heat curve. The report therefore includes a regularization audit:
best $\lambda$, gain relative to $\lambda=0$, gain divided by seed standard
error, and response-scale comparisons. Only if that audit shows a robust
regularization gain should $C_\lambda$ be promoted back to a phase-boundary
candidate.

## 3D View

The 3D MNIST phase view has two modes. The raw iso-loss mode is the literal
three-dimensional interpolation of validation cross-entropy over
$(\log_2 N,\log_2 W,\log_{10}\lambda)$. In this dense local run, the mean loss
variation across $N$ and $W$ is much larger than the mean variation across
$\lambda$, so raw iso-loss surfaces can look nearly planar in the regularization
direction.

The default balanced curvature residual mode subtracts the additive main
effects of $N$, $W$, and $\lambda$:

$$
\Delta R(N,W,\lambda)
=
R(N,W,\lambda)
-\bar R_N(N)
-\bar R_W(W)
-\bar R_\lambda(\lambda)
+2\bar R .
$$

This residual is not a free energy or a literal loss. It is a visualization
normalization that makes interaction curvature visible when one control's main
effect is numerically small. The $\lambda=0$ cells remain in the 2D phase plots
and summary tables, but are excluded from the 3D log-regularization axis.

The loss-decomposition plot is also diagnostic. It displays

$$
R_{\rm reported}
=
R_{\rm CE}
+
\frac{\lambda}{2}\lVert w\rVert_2^2 .
$$

Because training uses AdamW, whose weight decay is decoupled from the gradient
of an additive penalty, the regularizer bar is a measured diagnostic component,
not the exact optimized objective.

## Can We Estimate Free Energy?

Only in restricted senses.

For supervised learning, the literal statistical-mechanics free energy would be

$$
F(\beta)
=
-\frac{1}{\beta}\log
\int \exp[-\beta L_{\rm train}(w)]\,p_0(w)\,dw,
$$

or a constrained version with Lagrange fields. Accurately estimating this for a
neural network is hard: it requires a prior or measure over weights, a
temperature/noise model, and integration over basins rather than a single
optimizer trajectory.

The present report estimates an effective response potential $R(\theta)$, not
the true partition-function free energy. Its Legendre-like dual is empirical:

$$
\Gamma(m)
=
\sup_h \{hm - F(h)\},
$$

but this is exact only when the observable $m$ and field $h$ define a real
ensemble. For neural networks, the response curves are diagnostics until they
show invariance across seeds, architectures, and training protocols.

## Scaling Plan

The local dense run is a design probe. A useful remote sweep should be closer to

$$
N_{\rm runs}
=
N_{\rm seeds}\,
N_N\,
N_W\,
N_\lambda\,
N_\eta\,
N_T\,
N_{\rm arch}.
$$

A $10^5$-run MNIST study is feasible only if each run is very small, early
stopped, and checkpointed sparsely. The shape I would use is:

- 3-5 seeds;
- 20-30 clustered log-spaced data sizes, emphasizing $N\gtrsim 10^4$;
- 20-30 widths or parameter counts;
- 15-25 regularization values;
- 3-5 learning-rate or optimizer settings;
- a few training-time checkpoints or an early-stop envelope;
- no hysteresis in the main sweep.

This should be launched as resumable shards with one row per retrain and a
predeclared run budget. The first remote milestone should not be $10^5$ runs
immediately; it should be a $10^3$-$10^4$ shard that verifies derivative
stability and identifies whether any regularization-specific ridge survives
protocol and seed changes.

## Interpretation Rule

The words "data-limited", "capacity-limited", and "regularization-limited" are
local response labels:

$$
\operatorname*{argmax}_i |h_i(\theta)|.
$$

They mean "this perturbation most changes validation risk near this protocol."
They do not mean the model has a unique thermodynamic state variable.
