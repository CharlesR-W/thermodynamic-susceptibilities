# Thermodynamic Susceptibilities for Learning Systems

## One-Sentence Summary

This is a finite MNIST pilot for the idea that neural-network "capacity" can be
treated as a constrained resource, with Lagrange multipliers acting as
susceptibilities or shadow prices. The larger aim is to put heterogeneous
regularizers on a common footing: norm budgets, rank constraints, sparsity,
noise robustness, and architecture choices should all have comparable conjugate
fields if they can be written as softened constraints. The current run is
useful mainly because it is not a clean success: data and width responses are
visible, but the direct weight-decay response is too weak to justify a
phase-boundary claim.

## The Capacity Hypothesis

The intended mathematical object is a constrained risk problem. Let

$$
R(w)
=
\mathbb{E}_{(x,y)\sim \mathcal{D}_{\rm val}}
\left[\ell(f_w(x), y)\right]
$$

be validation risk, and let $q(w)$ be a resource or complexity observable. For
an L2-capacity story, the simplest choice is

$$
q(w)=\lVert w\rVert_2^2.
$$

Define the best achievable risk under a capacity budget $B$:

$$
R^*(B)=\min_{w:q(w)\le B} R(w).
$$

The Lagrangian is

$$
\mathcal{L}(w,\mu;B)=R(w)+\mu(q(w)-B),
$$

and, when the usual regularity conditions hold, the envelope theorem gives

$$
\frac{dR^*}{dB}=-\mu^*(B).
$$

So the multiplier $\mu^*$ is a shadow price of capacity: it measures how much
risk would fall if the norm budget were relaxed. This is the reason the
susceptibility language is interesting. A capacity definition is not just
"larger model good"; it is a response curve saying how much the task benefits
from marginally relaxing a constraint.

The limiting sanity check is also clear. If the constraint is genuinely an L2
capacity budget, then as $B\to 0$ the network should collapse toward a
near-zero or constant predictor. In that limit the task-specific capacity should
go to zero. Equivalently, an extremely large additive L2 penalty should destroy
the network's ability to fit MNIST.

## Why This Is More Than L2

The real motivation is that the multiplier language gives a level playing field
for regularization. For each proposed regularizer or architectural bottleneck,
choose an observable $q_a(w)$ and a budget $B_a$:

$$
\mathbb{E}[q_a]=B_a.
$$

Then the conjugate multiplier $\lambda_a$ says how expensive that constraint is
for the task. This suggests comparing constraints by response rather than by
their native units. Examples include:

- weight scale, such as $\lVert w\rVert_2^2$;
- spectral norm, nuclear norm, effective rank, or singular-value decay;
- activation sparsity, dead-unit fraction, or mask entropy;
- input noise, weight noise, dropout, or augmentation strength;
- architectural bottlenecks such as width, rank, attention head count, or
  low-rank adapters.

The hope is that different microscopic constraints may collapse onto a smaller
set of macroscopic response laws. That is the sense in which this could become
"thermodynamics": not a metaphor for one plot, but a search for universality
classes across constraint families.

## Ensembles And MaxEnt Softening

The more thermodynamic object is an ensemble of networks, not a single trained
network. Hard constraints can be softened into expectation constraints. For a
distribution $p(w)$ over networks, a maximum-entropy problem has the form

$$
\max_p H[p]
\quad\text{subject to}\quad
\mathbb{E}_p[L(w)]=E,\qquad
\mathbb{E}_p[q_a(w)]=B_a.
$$

The solution is a Gibbs family

$$
p(w)
\propto
\exp\left[-\beta L(w)-\sum_a \lambda_a q_a(w)\right],
$$

with

$$
\mathbb{E}_p[q_a(w)]=B_a.
$$

In this form, $\lambda_a$ is literally the field conjugate to observable $q_a$.
For $q(w)=\lVert w\rVert_2^2$, the max-entropy distribution with only an
expected norm constraint is Gaussian-like. With a loss term included, it becomes
a Gibbs posterior or regularized ensemble. This version may be analytically
tractable in local quadratic, NTK, Laplace, or other linearized regimes.

This is also the route for non-smooth constraints. "Rank equals $R$" is a hard
architectural condition, but nuclear norm, effective rank, spectral entropy, or
singular-value temperature are softened observables. They can have conjugate
fields and derivatives even when the original architecture switch is discrete.

## Penalties Versus Constraints

The penalty form

$$
\min_w R(w)+\frac{\lambda}{2}\lVert w\rVert_2^2
$$

is the standard Lagrangian proxy for the constrained problem. But interpreting
$\lambda$ as the multiplier is exact only when the training objective really is
the additive penalty and optimization is close enough to stationarity.

This report uses AdamW weight decay. AdamW is decoupled from the gradient of an
additive L2 term, so its weight-decay parameter is adjacent to the Lagrangian
story but not identical to it. The plotted diagnostic term

$$
\frac{\lambda}{2}\lVert w\rVert_2^2
$$

therefore helps check scale, but it is not the exact objective optimized during
training. This matters a lot for the "super-heavy L2 should zero capacity"
claim: that limiting claim requires either a true additive-L2 sweep or an
explicit norm-constrained solve, and a much wider regularization path than this
local run used.

Noise regularization fits the same template but with a different observable. If
input or weight perturbations are sampled from a noise distribution, the
effective risk is

$$
R_\sigma(w)
=
\mathbb{E}_{\epsilon}
\left[\ell(f_{w+\epsilon_w}(x+\epsilon_x),y)\right].
$$

The noise scale $\sigma$ is then a field controlling robustness. For small
noise, Taylor expansion turns this into curvature or Jacobian-like penalties,
which may be easier to linearize and compare with norm or rank constraints.

## Coordinate And Parameterization Caveats

The current report uses log-scale finite differences because "one more unit" of
data, width, or weight decay is not a comparable perturbation. This is a
practical normalization, not a principled invariant.

A more invariant response would differentiate with respect to the budget itself:

$$
\mu_a=-\frac{\partial R^*}{\partial B_a}.
$$

For scale-like budgets, a dimensionless elasticity may be more comparable:

$$
\chi_a
=
-\frac{\partial R^*}{\partial \log B_a}
=
B_a\mu_a.
$$

In the ensemble picture, the local geometry is supplied by the covariance matrix
of observables,

$$
g_{ab}=\operatorname{Cov}_p(q_a,q_b),
$$

which is also the Fisher information matrix of the exponential family. This is
a candidate way to compare constraint directions after whitening by their
natural fluctuations.

Neural networks add a further problem: many parameterizations represent the same
function. Raw L2 norm is not invariant under common symmetries such as ReLU
rescalings across layers. Better candidates may include path norms, spectral
quantities, function-space KL/Fisher distance, NTK-local coordinates, or
constraints on predictions and representations rather than raw weights. This is
one of the main theoretical issues a serious version of the program has to
solve.

## What This Run Actually Measures

The empirical potential in the report is validation cross-entropy

$$
R(N,W,\lambda),
$$

where

- $N$ is the number of MNIST training examples,
- $W$ is the hidden width of a one-hidden-layer ReLU MLP,
- $\lambda$ is the AdamW weight-decay parameter.

The normalized coordinates are

$$
\theta_N=\log_2 N,\qquad
\theta_P=\log_2 P,\qquad
\theta_\lambda=\log_{10}\lambda,
$$

where $P$ is the parameter count implied by width. The finite-difference fields
are

$$
h_N=-\frac{\partial R}{\partial\log_2 N},
\qquad
h_P=-\frac{\partial R}{\partial\log_2 P},
\qquad
h_\lambda=-\frac{\partial R}{\partial\log_{10}\lambda}.
$$

These have plain readings:

- $h_N>0$ means a data doubling improves validation loss.
- $h_P>0$ means a parameter/width doubling improves validation loss.
- $h_\lambda>0$ means stronger weight decay improves validation loss locally.
- $h_\lambda<0$ means stronger weight decay hurts validation loss locally.

The specific-heat-like diagnostic is a second finite difference,

$$
C_i=\left|\frac{\partial h_i}{\partial \theta_i}\right|
=
\left|\frac{\partial^2R}{\partial \theta_i^2}\right|.
$$

It asks where the usefulness of a control changes rapidly. It is also much more
noise-sensitive than $R$ or $h_i$, so it should not be treated as evidence unless
the underlying first derivative is already robust.

## What Would Count As Evidence

For the L2-capacity hypothesis, a stronger run should show at least four things:

- A monotone or interpretable degradation as the norm budget tightens, including
  a regime close to small-norm collapse.
- A regularization response $h_\lambda$ that is large compared with seed noise
  and not dwarfed by the data/width responses.
- A stable ridge or transition in $C_\lambda$ that survives seeds, interpolation
  choices, optimizer choices, and a predeclared thresholding rule.
- Agreement between the penalty path and the constrained-capacity interpretation,
  or an explicit reason why they can be compared.

The current run does not clear that bar. It shows useful data and width response
structure, but weight decay is a weak direction over the tested range.

## Protocol

The dense local MNIST run uses:

- scratch retrains;
- fixed 3-epoch exposure;
- one-hidden-layer ReLU MLPs on cached MNIST, downsampled to $14\times14$;
- fixed optimizer family and batch size;
- a clustered log grid over $N$, $W$, and $\lambda$;
- three independent scratch seeds per condition, with the sampled data subset
  held fixed.

The primary controls are approximately

$$
N\in[8192,50000],\qquad
W\in[32,256],\qquad
\lambda\in[10^{-6},3\times 10^{-2}],
$$

with $\lambda=0$ included as a reference cell but excluded from
$\log_{10}\lambda$ derivatives.

This is a local derivative estimator, not a thermodynamic limit and not a
complete constrained-capacity experiment.

## How To Read The Plots

The dense MNIST phase surface is the most literal figure. Each panel fixes
weight decay and shows validation cross-entropy over data size and width. If
the panels look similar as $\lambda$ changes, that is evidence that the tested
regularization range is not dominating the risk surface.

The regularization audit is the key honesty check. It shows the best
$\lambda$, the gain versus $\lambda=0$, the gain divided by seed standard error,
and the scale of $|h_\lambda|$ relative to $|h_N|$ and $|h_P|$. In this run,
that audit argues against a clean regularization phase-boundary story.

The lower-envelope plot is useful but post-hoc:

$$
R^*(N,W)=\min_\lambda R(N,W,\lambda).
$$

It asks what data or width still buys after weight decay has been tuned. This
is a reasonable descriptive object, but it should not be confused with
independent evidence that weight decay defines capacity.

The raw $\lambda$ finite-difference curves are intentionally demoted. They are
second derivatives of a small, noisy signal. They are included to show why an
attractive-looking bump should not be promoted into a discovered phase boundary.

The 3D residual view is also diagnostic. Its default mode subtracts additive
main effects of $N$, $W$, and $\lambda$:

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
normalization that makes small interactions visible after the dominant axis
trends have been removed.

## Can We Estimate Free Energy?

Only in restricted senses. A literal statistical-mechanics free energy would be

$$
F(\beta)
=
-\frac{1}{\beta}\log
\int \exp[-\beta L_{\rm train}(w)]\,p_0(w)\,dw,
$$

or a constrained version with Lagrange fields. Estimating this for a neural
network requires a prior or measure over weights, a temperature/noise model, and
integration over basins rather than one optimizer trajectory.

This report estimates an effective response potential $R(\theta)$, not a true
partition-function free energy. The susceptibility language is useful only if
the response fields remain stable across seeds, protocols, and architectures.

## Scaling Plan

The next experiment should be designed around the capacity claim rather than
around exploratory plot generation. A useful remote sweep should predeclare:

- whether the regularization axis is additive L2, decoupled AdamW, or an
  explicit norm constraint;
- the tested collapse criterion for small-norm/zero-capacity behavior;
- the threshold for treating a regularization response as real;
- which plots are confirmatory and which are exploratory diagnostics.

A practical remote design is closer to

$$
N_{\rm runs}
=
N_{\rm seeds}
N_N
N_W
N_\lambda
N_\eta
N_T
N_{\rm arch}.
$$

The first milestone should not be the full $10^5$ runs. It should be a
$10^3$-$10^4$ shard that verifies derivative stability and checks whether any
regularization-specific ridge survives seed, optimizer, and stopping-rule
changes.

## Interpretation Rule

The words "data-limited", "capacity-limited", and "regularization-limited" are
local response labels:

$$
\operatorname*{argmax}_i |h_i(\theta)|.
$$

They mean "this perturbation most changes validation risk near this protocol."
They do not mean the model has a unique thermodynamic state variable.
