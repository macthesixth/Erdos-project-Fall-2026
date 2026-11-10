"""Stand-alone, LOCAL PYTHON project: neural variational GBM delta hedging.

This file contains (1) the original GBM Method 2 variational inference functions
and (2) a causal synthetic option hedging experiment using those same functions.
It does NOT require Jupyter, Google Colab, or a second Python source file.

Install:  python -m pip install -r requirements.txt
Quick:    python GBM_Variational_Hedging_Local.py
Pilot:    python GBM_Variational_Hedging_Local.py --paths 8 --obs 3 --train-steps 200 --samples 1024 --particles 12000 --outdir pilot_results

The quick run checks execution only; it is NOT adequate for scientific inference.
All market observations are synthetic; trades use latent simulated execution
prices solely for controlled experiments, not an executable trading strategy.
"""
from __future__ import annotations

# ============================================================================
# HOW TO READ THIS FILE
# ============================================================================
# The script has TWO independent collections of functions:
#   PART 1: Your original research code: a normalizing-flow approximation to
#           posterior GBM marginal densities, plus an OPTIONAL EM procedure.
#   PART 2: An experiment that calls the PART 1 flow repeatedly and compares
#           different Black--Scholes delta-hedging rules on simulated prices.
#
# EXECUTED by the default command: main() -> sanity_checks() -> run_experiment().
# Not executed by the default command: run_em(), reference_paths(), _plots().
# In the main hedging experiment mu (the physical GBM drift) is treated as KNOWN.
# The OPTIONAL EM routine estimates drift for a separate inference experiment.
#
# The main scientific question is NOT whether the inferred drift changes the
# Black--Scholes formula. It is whether learning a distribution for an uncertain
# CURRENT stock price changes hedge decisions and their cost/risk trade-off.
#
# IMPORTANT: A separate flow is fitted at each observation prefix. For the
# hedge at t_k, the model sees only observations through t_k, NOT future data.
# ============================================================================
# ====== PART 1: ORIGINAL GBM METHOD 2 INFERENCE ======
"""
gbm_method2_em.py -- everything for GBM in one file.
=====================================================
Method 2 (flow-modelled posterior marginals) + EM inference of theta for

    dX = theta X dt + sigma X dW,    X(0) = x0

Work in log space, Y = ln X, where the SDE has CONSTANT coefficients

    dY = beta dt + sigma dW,   beta = theta - sigma^2/2,   a_Y = sigma^2

so a_Y never vanishes: no clamps, no positivity penalty, no x_min.

CONTENTS
    1  data              Euler-Maruyama latent path + noisy observations
    2  flow              piecewise-in-time monotone tanh flow (one block per
                         observation interval) + affine shift/scale
    3  control           u from the 1D CDF closed form
    4  loss              eq (5.6): energy + observations + anchor + continuity
    5  M-step            MARGINAL (default) and path-based (cross-check)
    6  EM                with window stopping rule and trace saving
    7  reference         backward-PDE smoother, for validating the marginals

--------------------------------------------------------------------------
THE M-STEP NEEDS NO PATH SIMULATION
--------------------------------------------------------------------------
The path M-step is beta_hat = E[Y_T - Y_0]/T.  Under the posterior SDE
dY = (beta_old + a_Y u) dt + sigma dW the martingale part has zero mean, so

    E[Y_T - Y_0] = INT_0^T E_q[beta_old + a_Y u(Y,t)] dt        (DRIFT form)
                 = E_{q(.,T)}[Y] - E_{q(.,0)}[Y]                (ENDPOINT form)

Both are expectations against marginals the flow already provides.  The drift
form is the general recipe (it carries over to the double well and to
Michaelis-Menten); the endpoint form is a GBM-only degeneracy -- the increment
telescopes because beta is constant -- and needs no control at all.

Measured on a trained flow at theta_old = 1.0 (see the docstring of run_em for
the setup):

    paths n_steps= 20 (dt=0.1000)      0.9991     0.2s
    paths n_steps=100 (dt=0.0200)      1.0129     0.6s
    paths n_steps=400 (dt=0.0050)      1.0121     2.6s      <- converged
    marginal ENDPOINT n=16384          1.0099     0.1s
    marginal DRIFT    n_time=200       1.0107     4.6s

    endpoint repeatability  1.0099 +- 0.0001
    drift    repeatability  1.0093 +- 0.0002
    path     repeatability  1.0124 +- 0.0002

All three agree to ~0.002, and the endpoint form is 26x faster than the
converged path simulation.  Keeping both marginal forms is useful: if they
disagree, u is inconsistent with the marginals it is supposed to transport.

(For GBM the Euler bias is mild -- 0.013 between dt=0.1 and dt=0.005 -- because
log-space GBM has constant coefficients.  In the double well, whose drift has
an x^3 term, the same comparison showed a bias of 0.64 at the default
n_steps=100, so there the marginal M-step is a correctness fix, not just a
speed-up.)

--------------------------------------------------------------------------
IDENTIFIABILITY -- READ BEFORE CHOOSING T
--------------------------------------------------------------------------
The M-step is exactly (ln X_T - ln x0)/T + sigma^2/2, whose error for ONE
latent realisation is sigma/sqrt(T).  With sigma = 0.2 and T = 0.4 that is
0.32, so EM cannot land near theta_true no matter how well the flow is
trained.  The default below (sigma = 0.1, T = 2.0) gives 0.07.  run_em prints
theta_oracle, the least-squares value from the TRUE path; EM should converge
there, and theta_oracle is close to theta_true only if T is long enough.
"""
import os, sys, math, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
if not hasattr(np, "trapz"): np.trapz = np.trapezoid
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

# The original inference code works with the LOG stock price Y = log(S).
# If dS = mu*S*dt + sigma*S*dW, Ito's formula gives
#   dY = beta*dt + sigma*dW, beta = mu - sigma**2/2.
# Working in log space makes the diffusion coefficient constant (a_Y=sigma**2),
# and exponentiating sampled Y guarantees positive stock prices.
# The hedging experiment below uses THIS fixed SIGMA for both data and pricing.
# ===================== model constants =====================
SIGMA = 0.1
SIG2 = SIGMA**2
A_Y = SIG2                       # log-space diffusion, constant
SQRT2 = math.sqrt(2.0)


# ============================ 1  data ============================
# OPTIONAL EM/PDE SIDE: approximate one latent GBM path using Euler-Maruyama.
# The main hedge experiment instead uses simulate_exact_gbm() below to avoid
# time-discretization bias in the synthetic benchmark.
def simulate_gbm_euler_maruyama(theta, x0, T, n_steps, seed):
    """Explicit Euler-Maruyama for dX = theta X dt + sigma X dW."""
    rng = np.random.default_rng(seed)
    dt = T/n_steps; sdt = math.sqrt(dt)
    t = np.linspace(0.0, T, n_steps+1)
    X = np.empty(n_steps+1); X[0] = x0
    for n in range(n_steps):
        X[n+1] = max(X[n] + theta*X[n]*dt + SIGMA*X[n]*sdt*rng.standard_normal(), 1e-8)
    return t, X, rng


# Observation model inherited from your code:
#   observed_price_k = latent_stock_price(t_k) + epsilon_k,
#   epsilon_k ~ Normal(0, R).
# NOTE R is a VARIANCE in price units (noise standard deviation = sqrt(R)).
# This is NOT Gaussian noise in log-price, and is why the main script uses a
# particle-filter baseline rather than claiming an exact Kalman-filter solution.
def make_observations(t_fine, X_fine, obs_times, R, rng):
    idx = [int(np.argmin(np.abs(t_fine - tk))) for tk in obs_times]
    xat = X_fine[idx]
    return xat + math.sqrt(R)*rng.standard_normal(len(obs_times)), xat


# ============================ 2  flow ============================
# Map normalized time tau into [tau, sin(2*pi*f*tau), cos(2*pi*f*tau), ...].
# These Fourier-like features help the flow change its density smoothly in time.
class TimeFeat(nn.Module):
    def __init__(self, nf=3):
        super().__init__()
        self.register_buffer("freqs", 2.0**torch.arange(nf, dtype=torch.float32))
        self.out_dim = 1 + 2*nf
    def forward(self, t):
        t = t.reshape(-1, 1); ang = 2*math.pi*t*self.freqs[None, :]
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], 1)


# One monotone 1D change-of-variables layer. The NN predicts time-dependent
# positive w,v and unrestricted b, yielding
#   z = y + w(t)*tanh(v(t)*y+b(t)).
# Since dz/dy = 1+w*v*sech^2(.) > 0, the inverse exists. The log-Jacobian
# is needed to compute q_t(y) by the normalizing-flow change-of-variables rule.
class Layer1D(nn.Module):
    """z = x + w(t) tanh(v(t) x + b(t)),  w,v > 0: monotone increasing."""
    def __init__(self, cdim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, 16), nn.Tanh(), nn.Linear(16, 3))
    def _wvb(self, c):
        p = self.net(c)
        return F.softplus(p[:, 0:1]), F.softplus(p[:, 1:2]), p[:, 2:3]
    def forward(self, x, c):
        w, v, b = self._wvb(c); th = torch.tanh(v*x + b)
        return x + w*th, torch.log(1 + w*v*(1 - th**2) + 1e-12)
    def inverse(self, z, c, n_iter=30):
        w, v, b = self._wvb(c); x = z.clone()
        for _ in range(n_iter):
            th = torch.tanh(v*x + b)
            x = x - (x + w*th - z)/(1 + w*v*(1 - th**2) + 1e-9)
        return x


# Additional affine transform to let the full distribution's location/width
# change substantially with time without requiring many tanh layers.
class ShiftScale(nn.Module):
    """Affine z = (y - mu(t))/s(t).  The tanh layers shift by O(w) and sharpen
    by O(1) each; on a long horizon the marginal has to move a long way from
    the near-delta at t=0, and this gives the flow that degree of freedom
    directly instead of spending layers on it."""
    def __init__(self, cdim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cdim, 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x, c):
        p = self.net(c); mu = p[:, 0:1]; ls = torch.clamp(p[:, 1:2], -6., 6.)
        return (x - mu)*torch.exp(-ls), -ls.expand_as(x)
    def inverse(self, z, c, n_iter=0):
        p = self.net(c); mu = p[:, 0:1]; ls = torch.clamp(p[:, 1:2], -6., 6.)
        return z*torch.exp(ls) + mu


# Construct one time-dependent density model per interval between observations.
# The forward map transforms log-price y -> standard-normal z; the inverse map
# transforms z ~ N(0,1) -> log-price samples y at any requested time.
# log_prob() evaluates q_t(y) via the normal Jacobian-change formula.
class Flow1D(nn.Module):
    """Piecewise in time: independent block on each (times[k], times[k+1]]."""
    def __init__(self, times, n_layers=4, nf=3):
        super().__init__()
        self.register_buffer("times", torch.as_tensor(times, dtype=torch.float32))
        self.K = int(self.times.numel()) - 1
        self.feat = nn.ModuleList([TimeFeat(nf) for _ in range(self.K)])
        self.layers = nn.ModuleList([
            nn.ModuleList([Layer1D(self.feat[k].out_dim) for _ in range(n_layers)]
                          + [ShiftScale(self.feat[k].out_dim)])
            for k in range(self.K)])
    def _interval(self, t):
        m = int(torch.bucketize(t.reshape(()), self.times, right=False) - 1)
        return max(0, min(m, self.K - 1))
    def _ctx(self, m, tv):
        tau = (tv - self.times[m])/(self.times[m+1] - self.times[m] + 1e-12)
        return self.feat[m](tau)
    def base(self, x, tv, interval=None):
        m = self._interval(tv[0]) if interval is None else int(interval)
        c = self._ctx(m, tv); z, ld = x, torch.zeros_like(x)
        for L in self.layers[m]:
            z, l = L(z, c); ld = ld + l
        return z, ld
    def log_prob(self, x, tv, interval=None):
        z, ld = self.base(x, tv, interval)
        return (-0.5*(z**2 + math.log(2*math.pi)) + ld).squeeze(1)
    def invert(self, zeta, tv, n_iter=30, interval=None):
        m = self._interval(tv[0]) if interval is None else int(interval)
        c = self._ctx(m, tv); x = zeta
        for L in reversed(self.layers[m]):
            x = L.inverse(x, c, n_iter)
        return x


# Replicate a scalar time t to the same batch/device/dtype as the state inputs.
def _tvec(t, N, like=None):
    if like is not None:
        return torch.full((N,), float(t), device=like.device, dtype=like.dtype)
    return torch.full((N,), float(t))


# Central sampling interface used throughout the project:
#   zeta_j ~ Normal(0,1),    Y_j(t) = flow^{-1}(zeta_j,t),
#   S_j(t) = exp(Y_j(t)).
# Newton iterations calculate the inverse NUMERICALLY. The final correction
# reintroduces the derivative using implicit differentiation, allowing loss
# gradients to flow through samples without backpropagating through Newton's
# entire iteration history.
def sample_Y(flow, zeta, t, interval=None):
    """Y = f_theta(zeta, t): value from Newton, gradient from the IFT."""
    tv = _tvec(t, zeta.shape[0], like=zeta)
    with torch.no_grad():
        Ys = flow.invert(zeta, tv, interval=interval)
    if not torch.is_grad_enabled():
        return Ys
    Ys = Ys.detach()
    z, ld = flow.base(Ys, tv, interval=interval)
    return Ys - (z - zeta)/torch.exp(ld)


# ============================ 3  control ============================
# Derive the control u(t,y) implied by the learned 1D marginal family q_t.
# Write F(t,y)=int_{-infinity}^y q_t(v)dv. Integrating the forward equation
# for the controlled diffusion dY=(beta+a_Y*u)dt+sigma*dW yields
#
#      a_Y*u*q = - partial_t F - beta*q + (a_Y/2)*partial_y q.
#
# The flow makes F=Phi(z(y,t)) easy to evaluate, and PyTorch autograd gives
# its time derivative and the spatial density derivative.
# This identity is the mathematical bridge between the neural marginal density
# and the path-space control-energy term in the variational objective.
def u_and_q(flow, Y, t, beta_y, create_graph=True):
    """
    1D CDF closed form (z = f^{-1} monotone, so Q = Phi(z)):
        a_Y u q = -d_t Q - beta_y q + (1/2) a_Y d_y q
    """
    tv = _tvec(t, Y.shape[0], like=Y).requires_grad_(True)
    Yr = Y if Y.requires_grad else Y.requires_grad_(True)
    z, ld = flow.base(Yr, tv)
    q = torch.exp((-0.5*(z**2 + math.log(2*math.pi)) + ld).squeeze(1))
    Q = 0.5*(1 + torch.erf(z.squeeze(1)/SQRT2))
    # retain_graph: the two grad calls share one graph, so with
    # create_graph=False the first would free it and the second would crash.
    dQ_dt = torch.autograd.grad(Q.sum(), tv, create_graph=create_graph,
                                retain_graph=True)[0]
    dq_dy = torch.autograd.grad(q.sum(), Yr, create_graph=create_graph)[0].squeeze(1)
    return (-dQ_dt - beta_y*q + 0.5*A_Y*dq_dy)/(A_Y*q + 1e-12), q


# ============================ 4  loss ============================
# Approximate the path-space KL/control-energy part of Method 2:
#    E_q [ 1/2 * integral_0^T sigma**2 * u(t,Y_t)**2 dt ].
# Here both time and states are sampled by Monte Carlo, interval by interval.
# IMPORTANT: The energy should be evaluated on samples from the LEARNED q_t.
def energy_mc(flow, obs_times, T, beta_y, batch_t=4, batch=64, eps=1e-5):
    dev = next(flow.parameters()).device
    energy = torch.zeros((), device=dev)
    bnd = [0.0] + list(obs_times) + ([T] if obs_times[-1] < T else [])
    for a, b in zip(bnd[:-1], bnd[1:]):
        L = b - a
        if L <= 0: continue
        acc = torch.zeros((), device=dev)
        for _ in range(batch_t):
            tk = (a + eps + torch.rand(()).item()*(L - 2*eps)) if L > 2*eps else 0.5*(a+b)
            Y = sample_Y(flow, torch.randn(batch, 1, device=dev), tk)
            u, _ = u_and_q(flow, Y, tk, beta_y)
            acc = acc + torch.mean(0.5*A_Y*u**2)
        energy = energy + L*acc/batch_t
    return energy


# Each time interval has separate neural parameters; without a coupling term,
# adjacent interval distributions might disagree at their common boundary.
# Matching the quantiles sampled from the same base Gaussian z encourages
# continuity of the model's marginals. This is a SOFT numerical penalty,
# not an exact continuity constraint.
def continuity_penalty(flow, batch=64, cont_sd=0.03):
    """
    Tie the per-interval blocks together.  The smoothing marginal is continuous
    in t, but the blocks are parameterised independently; without this term each
    interval decouples and sees only the observation at its own right endpoint,
    i.e. it ignores every earlier observation.  Equality of the monotone
    quantile maps is equivalent to equality of the two distributions.
    """
    dev = next(flow.parameters()).device
    pen = torch.zeros((), device=dev)
    for k in range(flow.K - 1):
        tb = float(flow.times[k+1])
        zeta = torch.randn(batch, 1, device=dev)
        Yl = sample_Y(flow, zeta, tb, interval=k)
        Yr = sample_Y(flow, zeta, tb, interval=k+1)
        pen = pen + torch.mean(0.5*((Yl - Yr)/cont_sd)**2)
    return pen


# Total optimization criterion used in the neural inference E-step:
#   loss = control energy
#        + sum_k E_{q_tk}[-log p(observed_k | S_tk)]
#        + lambda_anchor * initial-state penalty
#        + lambda_cont   * interval-boundary penalty.
# For additive Gaussian PRICE noise, the negative log likelihood is
#   (obs-S)**2/(2*R) + (1/2)*log(2*pi*R).
# The anchor enforces the known initial log price only approximately.
# Monitoring ONLY the total loss is insufficient; inspect all components.
def loss_eq56(flow, obs_times, obs_y, R, lnx0, T, beta_y, batch=64, batch_t=4,
              lam_anchor=5.0, anchor_sd=0.03, lam_cont=5.0, cont_sd=0.03):
    dev = next(flow.parameters()).device
    energy = energy_mc(flow, obs_times, T, beta_y, batch_t, batch)
    obs = torch.zeros((), device=dev)
    for tk, yk in zip(obs_times, obs_y):
        Xk = torch.exp(sample_Y(flow, torch.randn(batch, 1, device=dev), tk)).squeeze(1)
        obs = obs + torch.mean(0.5*(yk - Xk)**2/R + 0.5*math.log(2*math.pi*R))
    Y0 = sample_Y(flow, torch.randn(batch, 1, device=dev), 0.0).squeeze(1)
    anchor = torch.mean(0.5*((Y0 - lnx0)/anchor_sd)**2)
    cont = continuity_penalty(flow, batch, cont_sd)
    total = energy + obs + lam_anchor*anchor + lam_cont*cont
    return total, dict(loss=float(total), energy=float(energy), obs=float(obs),
                       anchor=float(anchor), cont=float(cont))


# Optimize the flow's NN weights with Adam. The learning-rate scheduler gradually
# lowers the step size, and clipping caps unstable gradients. The returned history
# records loss components; lower loss is NOT proof of a calibrated posterior.
def train(flow, obs_times, obs_y, R, x0, T, beta_y, steps, opt,
          lr=2e-3, lr_final=None, batch=64, batch_t=4, log_every=200):
    lnx0 = math.log(x0)
    for g in opt.param_groups: g["lr"] = lr
    sch = (torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(lr_final/lr)**(1/max(steps,1)))
           if lr_final else None)
    hist, t0 = [], time.time()
    for k in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        loss, s = loss_eq56(flow, obs_times, obs_y, R, lnx0, T, beta_y, batch, batch_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
        opt.step()
        if sch: sch.step()
        hist.append(s)
        if k == 1 or k % log_every == 0 or k == steps:
            print(f"    train {k:4d} | loss {s['loss']:+9.3f} energy {s['energy']:8.3f} "
                  f"obs {s['obs']:+8.3f} anchor {s['anchor']:6.3f} cont {s['cont']:6.3f}"
                  f" | {time.time()-t0:5.0f}s", flush=True)
    return hist


# ============================ 5  M-step ============================
@torch.no_grad()
# OPTIONAL separate EM parameter-estimation study (NOT used by main hedging run).
# Estimate physical drift theta from the fitted posterior endpoint log-prices:
#   theta_new = (E_q[log S_T]-E_q[log S_0])/T + sigma**2/2.
# It avoids resimulating full posterior trajectories for the M-step.
def theta_endpoint(flow, T, n_samp=16384):
    """MARGINAL, endpoint form: (E_q[Y_T] - E_q[Y_0])/T + sigma^2/2."""
    YT = sample_Y(flow, torch.randn(n_samp, 1), T).mean()
    Y0 = sample_Y(flow, torch.randn(n_samp, 1), 0.0).mean()
    return float((YT - Y0)/T + 0.5*SIG2)


# OPTIONAL cross-check for the EM M-step. Integrate the learned controlled drift
# beta_old + sigma**2*u in time under the learned marginal q_t. Ideally this
# agrees with theta_endpoint() when numerical training/control are consistent.
def theta_drift(flow, beta_old, T, n_time=200, n_samp=2000, eps=1e-4):
    """MARGINAL, drift form: (1/T) INT E_q[beta_old + a_Y u] dt + sigma^2/2."""
    edges = np.linspace(eps, T, n_time+1)
    ts = 0.5*(edges[:-1] + edges[1:]); dtw = edges[1] - edges[0]
    acc = 0.0
    for t in ts:
        Y = sample_Y(flow, torch.randn(n_samp, 1), float(t))
        u, _ = u_and_q(flow, Y, float(t), beta_old, create_graph=False)
        acc += float(torch.mean(beta_old + A_Y*torch.nan_to_num(u.detach())))*dtw
    return acc/T + 0.5*SIG2


# OPTIONAL posterior path visualization/check. Euler-simulate the diffusion with
# the learned control. Main hedging uses marginal samples DIRECTLY instead;
# consequently it does not need this expensive path simulation at every date.
def simulate_posterior_paths(flow, theta, T, n_steps, x0, n_paths=2000, seed=0):
    """Path E-step.  Kept for the trajectory plot and as a cross-check."""
    g = torch.Generator().manual_seed(seed)
    beta_y = theta - 0.5*SIG2
    dt = T/n_steps; sdt = math.sqrt(dt)
    Y = torch.zeros(n_paths, n_steps+1); Y[:, 0] = math.log(x0)
    umax = 50.0/A_Y            # cap on the DRIFT a_Y*u, not on u itself
    for n in range(n_steps):
        # right-limit: u jumps at observation times, so a step starting at t_k
        # must use the interval to the RIGHT of t_k
        t = min(n*dt + 1e-6, T)
        Yn = Y[:, n].detach().clone().reshape(-1, 1).requires_grad_(True)
        with torch.enable_grad():
            u, _ = u_and_q(flow, Yn, t, beta_y, create_graph=False)
        u = torch.clamp(torch.nan_to_num(u), -umax, umax).detach()
        Y[:, n+1] = Y[:, n] + (beta_y + A_Y*u)*dt + SIGMA*torch.randn(n_paths, generator=g)*sdt
    return np.linspace(0, T, n_steps+1), torch.exp(Y).detach(), Y.detach()


# OPTIONAL estimate theta by averaging all log-price increments of full paths.
# Used as a consistency check for endpoint/drift EM estimates.
def theta_from_paths(Y_paths, dt):
    Y = Y_paths.detach().cpu().numpy() if torch.is_tensor(Y_paths) else np.asarray(Y_paths)
    return float(np.diff(Y, axis=1).mean()/dt + 0.5*SIG2)


# ============================ 6  EM ============================
#def convergence_state(trace, tol=0.02, patience=4, min_iters=0):
#    """'theta within tol for `patience` iterations', window form: all of the
#    last patience+1 iterates inside a band of width tol.  The successive-
#    difference form is fooled by a slow steady drift; this one is not."""
#    th = np.asarray(trace, float)
#    if len(th) < patience+1: return 0, False, "n/a"
#    w = th[-(patience+1):]
#    spread = float(w.max() - w.min())
#    return (patience if spread < tol else 0), \
#           (spread < tol and len(th)-1 >= min_iters), \
#           f"spread {spread:.4f} over last {patience+1}"


# OPTIONAL, DISTINCT EXPERIMENT: alternate between fitting a full-data posterior
# smoother (E-step) and updating the physical GBM drift (M-step).
# Do not confuse this with causal hedging: the EM posterior uses all available
# observations across its horizon and is intended for offline parameter inference.
# main() NEVER invokes run_em(); the hedging experiment fixes mu=0.08.
def run_em(n_iters=30, T=2.0, x0=1.0, R=0.04, n_obs=5, theta_true=1.0,
           theta_init=3.0, data_seed=12, mstep="endpoint",
           train_first=700, train_next=250, lr=3e-3, batch=64, batch_t=4,
           n_samp=16384, stop_tol=0.02, stop_patience=4, min_iters=4,
           cross_check=True, prefix="gbm"):
    torch.manual_seed(0); np.random.seed(0)
    obs_times = [round(T*(k+1)/n_obs, 6) for k in range(n_obs)]
    t_fine, X_fine, rng = simulate_gbm_euler_maruyama(theta_true, x0, T, 4000, data_seed)
    obs_y, x_at = make_observations(t_fine, X_fine, obs_times, R, rng)
    theta_oracle = (math.log(X_fine[-1]) - math.log(x0))/T + 0.5*SIG2

    print("="*78)
    print(f"GBM Method 2 + EM   theta_true={theta_true}  sigma={SIGMA}  x0={x0}  T={T}")
    print(f"  {n_obs} observations, R={R} (sd {math.sqrt(R):.3f}); init theta={theta_init}")
    print(f"  theta_oracle = {theta_oracle:.4f}  <- best possible from this latent path")
    print(f"  statistical floor sigma/sqrt(T) = {SIGMA/math.sqrt(T):.4f}")
    print(f"  M-step: {mstep} (marginal; no path simulation)")
    print("="*78, flush=True)

    flow = Flow1D([0.0] + obs_times, n_layers=4, nf=3)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    theta = float(theta_init); trace = [theta]; all_hist = []

    for it in range(n_iters):
        beta_y = theta - 0.5*SIG2
        steps = train_first if it == 0 else train_next
        this_lr = lr if it == 0 else lr*0.5
        print(f"\nEM iter {it}: theta={theta:.5f}", flush=True)
        all_hist += train(flow, obs_times, obs_y, R, x0, T, beta_y, steps, opt,
                          lr=this_lr, lr_final=this_lr*0.3, batch=batch,
                          batch_t=batch_t, log_every=max(1, steps//2))

        if mstep == "endpoint":
            th_hat = theta_endpoint(flow, T, n_samp)
        elif mstep == "drift":
            th_hat = theta_drift(flow, beta_y, T)
        else:
            _, _, Yp = simulate_posterior_paths(flow, theta, T, 200, x0, 2000)
            th_hat = theta_from_paths(Yp, T/200)

        extra = ""
        if cross_check:
            other = theta_drift(flow, beta_y, T, n_time=100, n_samp=2000)
            _, _, Yp = simulate_posterior_paths(flow, theta, T, 200, x0, 1000)
            extra = (f" | drift {other:.4f} paths {theta_from_paths(Yp, T/200):.4f}")

        theta = th_hat; trace.append(theta)
#        streak, stop, detail = convergence_state(trace, stop_tol, stop_patience, min_iters)
#        print(f"EM iter {it}: theta_hat={th_hat:.5f}{extra} | {detail}", flush=True)
#        if stop:
#            print(f"\n*** STOPPED at iteration {it}: theta stable within {stop_tol} "
#                  f"for {stop_patience} consecutive iterations.")
#            break

    trace = np.asarray(trace)
    np.savetxt(f"{prefix}_theta_trace.csv",
               np.column_stack([np.arange(len(trace)), trace]),
               delimiter=",", header="iter,theta", comments="", fmt="%.6f")
    np.savez(f"{prefix}_trace.npz", trace=trace, obs_times=obs_times, obs_y=obs_y,
             t_fine=t_fine, X_fine=X_fine, oracle=theta_oracle)
    torch.save({"flow": flow.state_dict(), "trace": trace}, f"{prefix}_ckpt.pt")

    print("\n" + "="*78)
    print("trace:", [round(float(t), 4) for t in trace])
    print(f"  theta_true   = {theta_true:.4f}")
    print(f"  theta_oracle = {theta_oracle:.4f}")
    print(f"  theta_EM     = {trace[-1]:.4f}  (last)   {trace[-4:].mean():.4f}  (mean last 4)")
    print(f"  |EM - oracle| = {abs(trace[-4:].mean()-theta_oracle):.4f}")
    print("="*78)

    _plots(flow, trace, theta_true, theta_oracle, obs_times, obs_y, t_fine,
           X_fine, T, x0, R, all_hist, prefix)
    return trace, flow, (t_fine, X_fine, obs_times, obs_y, theta_oracle)


# ============================ 7  reference + plots ============================
# OPTIONAL PDE REFERENCE for offline smoothing. This solves a discretized
# backward equation with observation-likelihood updates in price space.
# It is NOT the online reference used in the main hedging experiment.
def _backward_w(theta, y_obs, R_obs, t_obs, x, t_grid):
    dx, dt = x[1]-x[0], t_grid[1]-t_grid[0]
    Nt, Nx = len(t_grid)-1, len(x)
    w = np.zeros((Nt+1, Nx)); w[-1] = 1.0
    b = theta*x; ah = 0.5*SIG2*x**2
    lo = -dt*(ah/dx**2 - b/(2*dx)); di = 1.0 + dt*(2*ah/dx**2); up = -dt*(ah/dx**2 + b/(2*dx))
    A = diags([lo[1:], di, up[:-1]], [-1, 0, 1], shape=(Nx, Nx)).tocsc()
    def upd(wv, yk, Rk):
        lu = -0.5*(yk - x)**2/Rk; wv = wv*np.exp(lu - lu.max())
        Z = np.trapz(wv, x); return wv/Z if (Z > 0 and np.isfinite(Z)) else wv
    j = len(t_obs)-1
    if j >= 0 and np.isclose(t_grid[-1], t_obs[j], atol=dt/2):
        w[-1] = upd(w[-1], y_obs[j], R_obs[j]); j -= 1
    for n in reversed(range(Nt)):
        wp = spsolve(A, w[n+1].copy())
        if j >= 0 and np.isclose(t_grid[n], t_obs[j], atol=dt/2):
            wp = upd(wp, y_obs[j], R_obs[j]); j -= 1
        w[n] = wp
    return w


# OPTIONAL: simulate reference posterior trajectories using the backward PDE
# correction to the stock-price drift. Computationally costly; diagnostic only.
def reference_paths(x0, obs_times, obs_y, R, t_grid, theta,
                    x_lo=0.2, x_hi=15.0, Nx=600, n_paths=20000, seed=123):
    x = np.linspace(x_lo, x_hi, Nx)
    w = _backward_w(theta, np.asarray(obs_y), np.full(len(obs_times), R),
                    np.asarray(obs_times), x, t_grid)
    lw = np.log(np.maximum(w, 1e-300)); dx = x[1]-x[0]
    dlw = np.zeros_like(lw)
    dlw[:, 1:-1] = (lw[:, 2:] - lw[:, :-2])/(2*dx)
    dlw[:, 0] = (lw[:, 1]-lw[:, 0])/dx; dlw[:, -1] = (lw[:, -1]-lw[:, -2])/dx
    itp = [interp1d(x, dlw[n], kind="linear", fill_value="extrapolate")
           for n in range(len(t_grid))]
    rng = np.random.default_rng(seed); dt = t_grid[1]-t_grid[0]
    P = np.full((n_paths, len(t_grid)), float(x0))
    for n in range(len(t_grid)-1):
        Xn = P[:, n]
        sc = np.clip(itp[n](np.clip(Xn, x[0], x[-1])), -100, 100)
        P[:, n+1] = np.clip(Xn + (theta*Xn + SIG2*Xn**2*sc)*dt
                            + SIGMA*Xn*np.sqrt(dt)*rng.standard_normal(n_paths),
                            x[0], x[-1])
    return x, P


@torch.no_grad()
# Convert the flow density for Y=log(S) into a stock PRICE density:
#   q_S(s,t) = q_Y(log(s),t) / s,    s>0.
# The /s is the Jacobian of the log transform. The numerical normalization
# helps account for the finite x_grid used in plotting.
def q_theta_x(flow, x_grid, t):
    """q^X(x,t) = q^Y(ln x,t)/x."""
    Y = torch.tensor(np.log(x_grid), dtype=torch.float32).reshape(-1, 1)
    qX = torch.exp(flow.log_prob(Y, _tvec(t, Y.shape[0], like=Y))).numpy()/x_grid
    Z = np.trapz(qX, x_grid)
    return qX/Z if Z > 0 else qX


# OPTIONAL offline EM graphics: drift trace, controlled posterior paths,
# smoother-vs-flow density comparisons, and NN loss-component history.
def _plots(flow, trace, theta_true, theta_oracle, obs_times, obs_y, t_fine,
           X_fine, T, x0, R, hist, prefix):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trace, "o-", label="EM estimate")
    ax.axhline(theta_true, ls="--", color="k", label=r"true $\theta$")
    ax.axhline(theta_oracle, ls=":", color="#dd6b20", label=r"oracle $\theta$")
    ax.set_xlabel("EM iteration"); ax.set_ylabel(r"$\theta$")
    ax.grid(alpha=.3); ax.legend(); ax.set_title("GBM EM")
    fig.tight_layout(); fig.savefig(f"{prefix}_theta_trace.png", dpi=150); plt.close(fig)

    tp, Xp, _ = simulate_posterior_paths(flow, trace[-1], T, 200, x0, 50, seed=999)
    Xn = Xp.numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in range(Xn.shape[0]): ax.plot(tp, Xn[p], alpha=.3, lw=.9, color="#2b6cb0")
    ax.plot(t_fine, X_fine, color="k", lw=1.6, label="true latent path")
    ax.scatter(obs_times, obs_y, marker="x", s=60, color="#e53e3e", zorder=5,
               label="observations")
    ax.set_xlabel("t"); ax.set_ylabel("X"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("posterior trajectories from the Method 2 control")
    fig.tight_layout(); fig.savefig(f"{prefix}_paths.png", dpi=150); plt.close(fig)

    try:
        grid = np.round(np.arange(0.0, T + 1e-9, 0.005), 6)
        _, P = reference_paths(x0, obs_times, obs_y, R, grid, theta_true)
        dtg = grid[1]-grid[0]
        fig, axes = plt.subplots(1, len(obs_times), figsize=(3.4*len(obs_times), 3.4))
        for ax, tk in zip(np.atleast_1d(axes), obs_times):
            s = P[:, int(round(tk/dtg))]
            xg = np.linspace(max(1e-3, s.min()*0.85), s.max()*1.15, 200)
            ax.plot(xg, gaussian_kde(s)(xg), lw=2.1, color="#2b6cb0", label="backward-PDE")
            ax.plot(xg, q_theta_x(flow, xg, tk), "--", lw=2.1, color="#dd6b20",
                    label=r"Method 2 $q_\theta$")
            ax.axvline(obs_y[obs_times.index(tk)], ls=":", color="#e53e3e", lw=1.2)
            ax.set_title(f"t = {tk:.2f}"); ax.set_xlabel("x"); ax.grid(alpha=.3)
        np.atleast_1d(axes)[0].set_ylabel("density")
        np.atleast_1d(axes)[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{prefix}_marginals.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("PDE reference skipped:", e)

    if hist:
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in ("loss", "energy", "obs", "anchor", "cont"):
            ax.plot([h[k] for h in hist], lw=.9, label=k)
        ax.set_yscale("symlog"); ax.set_xlabel("training step (all EM iterations)")
        ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("eq (5.6) loss")
        fig.tight_layout(); fig.savefig(f"{prefix}_loss.png", dpi=150); plt.close(fig)
    print(f"saved {prefix}_theta_trace.csv/.png, {prefix}_trace.npz, {prefix}_ckpt.pt, "
          f"{prefix}_paths.png, {prefix}_marginals.png, {prefix}_loss.png")


# OPTIONAL consistency benchmark: compare drift estimates from simulated
# posterior paths, endpoint posterior marginals, and integrated controlled drift.
def compare_mstep(flow, theta_old, T, x0):
    """Path vs the two marginal forms, on one flow.  They should agree."""
    beta_y = theta_old - 0.5*SIG2
    print("  method                             theta_hat    time")
    for ns in (20, 50, 100, 200, 400):
        t0 = time.time()
        tp, _, Yp = simulate_posterior_paths(flow, theta_old, T, ns, x0, 2000, seed=7)
        print(f"  paths n_steps={ns:4d} (dt={T/ns:.4f})    "
              f"{theta_from_paths(Yp, tp[1]-tp[0]):8.4f}  {time.time()-t0:5.1f}s", flush=True)
    t0 = time.time(); v = theta_endpoint(flow, T)
    print(f"  marginal ENDPOINT                 {v:8.4f}  {time.time()-t0:5.1f}s")
    for nt in (50, 200):
        t0 = time.time(); v = theta_drift(flow, beta_y, T, n_time=nt)
        print(f"  marginal DRIFT n_time={nt:4d}       {v:8.4f}  {time.time()-t0:5.1f}s")


def _cli_iters(default=12):
    for a in sys.argv[1:]:
        if a.lstrip("+-").isdigit():
            return int(a)
    return default

# ============================================================================
# MAIN FINANCE EXPERIMENT (this is what runs from the command line)
# ============================================================================
# 1) Generate EXACT synthetic latent GBM paths with physical drift mu=0.08.
# 2) Observe only selected times, with additive Gaussian noise in PRICE.
# 3) At each observation t_k, run two CAUSAL inference methods:
#       a. Bootstrap particle filter (PF)  [independent reference]
#       b. Your neural Method 2 flow (VI) [main research contribution]
# 4) Compute Black--Scholes deltas, either at a point estimate or averaged
#    across posterior state samples; optionally apply an uncertainty threshold.
# 5) Compare the profits/losses of a short European-call hedged portfolio.
# 6) Export CSV summaries and diagnostic figures.
#
# Assumptions deliberately simplified: known mu and sigma, constant r, no
# dividends, synthetic execution at latent S, proportional transaction costs.
# The latent price must NOT enter the inference or non-oracle hedge decisions.
# The "Oracle diagnostic" DOES see latent prices and is NOT deployable.
# ============================================================================
# ====== PART 2: SYNTHETIC GBM OPTION HEDGING ======

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from scipy.special import ndtr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# --------------------------- Black-Scholes ---------------------------

# Black--Scholes EUROPEAN call premium:
#   C = S*exp(-q*tau)*Phi(d1) - K*exp(-r*tau)*Phi(d2),
#   d1=[log(S/K)+(r-q+sigma^2/2)*tau]/(sigma*sqrt(tau)),
#   d2=d1-sigma*sqrt(tau).
# Here r is the RISK-NEUTRAL pricing input, not the physical data drift mu.
# At tau=0 return payoff max(S-K,0). The main project sets dividend q=0.
def bs_call(S, K, r, sigma, tau, dividend=0.0):
    """B-S European call value, allowing scalar or array S and tau."""
    S = np.asarray(S, float)
    tau = np.asarray(tau, float)
    S, tau = np.broadcast_arrays(S, tau)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    out = np.maximum(S - K, 0.0)
    mask = tau > 0
    if np.any(mask):
        ss = np.maximum(S[mask], 1e-12)
        tt = tau[mask]
        d1 = (np.log(ss / K) + (r - dividend + .5 * sigma**2)*tt)/(sigma*np.sqrt(tt))
        d2 = d1 - sigma*np.sqrt(tt)
        out = np.asarray(out).copy()
        out[mask] = ss*np.exp(-dividend*tt)*ndtr(d1) - K*np.exp(-r*tt)*ndtr(d2)
    return float(out) if out.ndim == 0 else out


# Black--Scholes call delta dC/dS = exp(-q*tau)*Phi(d1), for tau>0.
# This is the number of SHARES held per short call to hedge price exposure.
# At expiry the delta is 1/0 depending on in/out-of-the-money; value 1/2
# at-the-money is a convention for a nondifferentiable terminal payoff.
def bs_delta(S, K, r, sigma, tau, dividend=0.0):
    """B-S European-call delta with standard expiry convention."""
    S = np.asarray(S, float)
    tau = np.asarray(tau, float)
    S, tau = np.broadcast_arrays(S, tau)
    out = np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
    mask = tau > 0
    if np.any(mask):
        ss = np.maximum(S[mask], 1e-12)
        tt = tau[mask]
        d1 = (np.log(ss / K) + (r - dividend + .5*sigma**2)*tt)/(sigma*np.sqrt(tt))
        out = np.asarray(out).copy()
        out[mask] = np.exp(-dividend*tt)*ndtr(d1)
    return float(out) if out.ndim == 0 else out


# --------------------------- synthetic data ---------------------------

# Synthetic DATA GENERATOR (PHYSICAL measure):
# S_{t+dt}=S_t*exp((mu-sigma^2/2)*dt+sigma*sqrt(dt)*Z).
# Exact GBM transition allows observed hedging errors to come from discrete
# hedging/observation noise, not from an Euler approximation to the SDE.
def simulate_exact_gbm(n_paths, x0, mu, sigma, T, n_steps, rng):
    """Exact solution, avoiding Euler discretization error in the test oracle."""
    t = np.linspace(0, T, n_steps + 1)
    z = rng.standard_normal((n_paths, n_steps))
    dy = (mu-.5*sigma*sigma)*(T/n_steps) + sigma*np.sqrt(T/n_steps)*z
    prices = x0 * np.exp(np.column_stack([np.zeros(n_paths), np.cumsum(dy, axis=1)]))
    return t, prices


# Particle-filter helper: duplicate high-weight particles and discard
# low-weight particles when the effective sample size gets too small.
def systematic_resample(weights, rng):
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    ids = np.searchsorted(np.cumsum(weights), positions, side="left")
    return np.minimum(ids, n-1)


# CAUSAL INDEPENDENT BENCHMARK for q(S_t | observations up to t):
#   a) Propagate particles using exact GBM transitions.
#   b) Multiply weights by Gaussian price-observation likelihood.
#   c) Compute posterior E[S], sd(S), E[Delta_BS(S)], and Delta_BS(E[S]).
#   d) Resample if effective sample size ESS falls below half of n_particles.
# The reference is approximate Monte Carlo, not an exact posterior formula.
def particle_filter_at_obs(observations, obs_times, *, x0, mu, sigma, R,
                           K, r, T, n_particles=12000, seed=0):
    """Price-noise bootstrap filter, causal; outputs posterior moments and deltas.

    State transitions are exact log-GBM; likelihood N(obs; S, R).
    Resampling is ESS-triggered. This is an approximate independent benchmark.
    """
    rng = np.random.default_rng(seed)
    s = np.full(n_particles, x0, dtype=float)
    weights = np.ones(n_particles)/n_particles
    t_old = 0.0
    outputs = []
    for tk, obs in zip(obs_times, observations):
        dt = float(tk-t_old)
        s *= np.exp((mu-.5*sigma**2)*dt + sigma*np.sqrt(dt)*rng.standard_normal(n_particles))
        # Bayes update: log weight += log p(observation | stock particle).
        # Gaussian normalizing constant cancels when weights are normalized.
        logw = np.log(np.maximum(weights, 1e-300)) -.5*(s-obs)**2/R
        logw -= np.max(logw)
        weights = np.exp(logw)
        weights /= np.sum(weights)
        mean = float(np.dot(weights, s))
        var = float(np.dot(weights, (s-mean)**2))
        davg = float(np.dot(weights, bs_delta(s, K, r, sigma, T-tk)))
        outputs.append({"mean":mean, "sd":math.sqrt(max(0, var)),
                        "delta_avg":davg,
                        "delta_plugin":float(bs_delta(mean, K, r, sigma, T-tk)),
                        "ess":float(1.0/np.sum(weights**2))})
        if outputs[-1]["ess"] < n_particles/2:
            s = s[systematic_resample(weights, rng)].copy()
            weights.fill(1.0/n_particles)
        t_old = float(tk)
    return outputs


# ------------------------ user's Method 2 ---------------------------

# THE CRUCIAL INTERFACE between your inference code and option hedging.
# When deciding a hedge at time t_k:
#   - build Flow1D only through t_k;
#   - fit Method 2 with observations 1,...,k (NO FUTURE LEAKAGE);
#   - sample log-prices Y_j ~ learned posterior q_{t_k};
#   - exponentiate S_j = exp(Y_j);
#   - output E[S], sd(S), Delta_BS(E[S]), and E[Delta_BS(S)].
# Previous interval weights are warm-started; new interval begins fresh.
# The previous run's flow is re-fit rather than an exact Bayesian update.
# No parameter EM is performed here: mu is fixed in advance.
def train_prefix_flows(observations, obs_times, *, x0, mu, R, steps,
                       batch, batch_t, n_samples, K, r, T, seed=0,
                       print_training=False):
    """Re-fit the user's 1D Method-2 flow for every OBSERVATION PREFIX.

    Uses original Flow1D and train / sample_Y; optimizer is
    reinitialized each prefix, earlier learned interval blocks are warm-started.
    mu is known and fixed for the primary hedging experiment; optional offline
    EM parameter inference remains in gbm_method2_em.py.
    """
    obs_times = np.asarray(obs_times, float)
    observations = np.asarray(observations, float)
    if len(obs_times) != len(observations):
        raise ValueError("observations and observation times must align")
    torch.manual_seed(seed)
    # FIXED physical log-price drift for the SDE PRIOR; not Black--Scholes r.
    beta = mu - .5*SIG2
    prev = None
    outputs = []
    timings = []
    # Online/causal evaluation: each new hedge sees an increasingly long PREFIX
    # of the observations, and none after the current decision time.
    for k in range(1, len(obs_times)+1):
        tk = float(obs_times[k-1])
        # Restrict the flow's time-domain to [0,t_k]. This is the key no-lookahead step.
        current_times = [0.0]+[float(v) for v in obs_times[:k]]
        flow = Flow1D(current_times, n_layers=4, nf=3)
        # Warm start matching interval blocks, including the time-feature layers.
        if prev is not None:
            for j in range(k-1):
                flow.layers[j].load_state_dict(prev.layers[j].state_dict())
        optimizer = torch.optim.Adam(flow.parameters(), lr=2e-3)
        start = time.perf_counter()
        # Fit neural posterior marginals using data through time t_k only.
        hist = train(flow, obs_times[:k].tolist(), observations[:k].tolist(),
                          R, x0, tk, beta, steps=steps, opt=optimizer,
                          lr=2e-3, lr_final=6e-4, batch=batch, batch_t=batch_t,
                          log_every=steps+1 if not print_training else max(1,steps//2))
        duration = time.perf_counter()-start
        timings.append(duration)
        with torch.no_grad():
            log_s = sample_Y(flow, torch.randn(n_samples, 1), tk).flatten().numpy()
        s = np.exp(np.clip(log_s, -25, 25))
        mean = float(s.mean())
        sd = float(s.std(ddof=1))
        # Average deltas AFTER applying the nonlinear delta function to samples.
        # Generally E[Delta(S)] != Delta(E[S]); testing this is a project question.
        delta_avg = float(np.mean(bs_delta(s, K, r, SIGMA, T-tk)))
        # Plug-in method discards posterior width, retaining just the mean price.
        delta_plugin = float(bs_delta(mean, K, r, SIGMA, T-tk))
        outputs.append({"mean":mean, "sd":sd, "delta_avg":delta_avg,
                        "delta_plugin":delta_plugin, "training_sec":duration,
                        "loss_last":hist[-1]["loss"],
                        "energy_last":hist[-1]["energy"],
                        "obs_last":hist[-1]["obs"]})
        prev = flow
        print(f"  prefix {k}/{len(obs_times)} at t={tk:.3f}: "
              f"mean={mean:.4f} sd={sd:.4f} VI-delta={delta_avg:.4f} "
              f"train={duration:.1f}s", flush=True)
    return outputs, prev, sum(timings)


# -------------------------- self-financing --------------------------

# ACCOUNTING ENGINE, for a SHORT EUROPEAN CALL plus its stock hedge.
# At time zero:
#   receive option premium C_0, buy h_0 shares at S_0,
#   cash_0=C_0-h_0*S_0-initial_transaction_cost.
# Between decisions cash grows at exp(r*dt), including negative cash balances
# (borrowing at the same r: a simplifying model assumption).
# At rebalance t_k, buying dh shares costs dh*S_t + cost*abs(dh)*S_t.
# At expiry, liquidate shares and pay call payoff (S_T-K)^+.
# Final P&L = cash_T + h*S_T - payoff - liquidation_cost.
#
# IMPORTANT CONTROLLED-EXPERIMENT ASSUMPTION: trades use the simulator's
# latent prices for EXECUTION, but non-oracle strategies choose hedge shares
# solely from noisy observations and inferred causal posteriors. This does
# not model real bid/ask execution under a genuinely unobserved price.
def hedge_one_path(prices, times, strike, r, sigma, T, target_deltas,
                   trade_cost=0., adaptive=False, cv=None,
                   delta0=.02, lam=.5):
    """Self-financing short-call delta hedge on simulated execution prices.

    times = [0, observation dates..., maturity]. target_deltas contains hedge
    targets for [0, observation dates...] (no rebalance at maturity).
    Transaction costs proportional to traded notional, including liquidation.
    The cash account earns continuously compounded r between rebalances.
    """
    prices = np.asarray(prices, float)
    times = np.asarray(times, float)
    target_deltas = np.asarray(target_deltas, float)
    if len(prices) != len(times) or len(target_deltas) != len(times)-1:
        raise ValueError("Inconsistent execution dates and hedge targets")
    if cv is None:
        cv = np.zeros(len(target_deltas))
    cv = np.asarray(cv,float)
    if not(np.all(np.diff(times)>0) and np.all(np.isfinite(target_deltas))):
        raise ValueError("Times must be increasing and deltas finite")
    h = float(target_deltas[0])
    # Receive European call premium, buy h shares; initial cost is included.
    cash = float(bs_call(prices[0], strike, r, sigma, T-times[0])) - h*prices[0]
    tc = trade_cost*abs(h)*prices[0]
    cash -= tc
    n_rebalances = 0
    held = [h]
    for k in range(1, len(times)-1):
        cash *= math.exp(r*(times[k]-times[k-1]))
        target = float(target_deltas[k])
        # Proposal: widen the no-trade region as uncertainty (posterior CV) rises.
        # If the desired adjustment is too small, hold existing shares to save fees.
        if adaptive and abs(target-h) <= delta0 + lam*cv[k]:
            held.append(h)
            continue
        dh = target-h
        if abs(dh) > 1e-12:
            transaction = trade_cost*prices[k]*abs(dh)
            cash -= dh*prices[k] + transaction
            tc += transaction
            n_rebalances += 1
            h = target
        held.append(h)
    cash *= math.exp(r*(times[-1]-times[-2]))
    liquidation = trade_cost*abs(h)*prices[-1]
    tc += liquidation
    pnl = cash + h*prices[-1] - max(prices[-1]-strike,0.) - liquidation
    return {"pnl":float(pnl),"cost":float(tc),
            "rebalances":n_rebalances,"positions":held}


# CVaR/expected shortfall of LOSS=-P&L. For small simulation counts this is
# extremely noisy; 20 paths are not enough to establish reliable 95% tail risk.
def cvar_loss(pnl, confidence=.95):
    """Empirical 95% CVaR of LOSS = -P&L; descriptive for small sample sizes."""
    losses = -np.asarray(pnl, float)
    threshold = np.quantile(losses, confidence)
    return float(losses[losses >= threshold].mean())


# ----------------------- run experiments ---------------------------

# MASTER WORKFLOW: construct synthetic data; perform online PF and VI inference;
# calculate hedge targets; run cash-account hedging; save CSVs and PNG figures.
# Default option is at-the-money (S0=K=1), maturity T=.8 years, volatility .1.
# This is a synthetic research comparison, NOT live-trading advice or evidence
# of profitability; the option premium comes from the same Black--Scholes model.
# The n_paths x n_obs x training_steps cost dominates runtime.
def run_experiment(n_paths=2, n_obs=3, epochs=160, n_samples=1024,
                   n_particles=12000, seed=123, outdir="gbm_hedging_results",
                   R=.0009, cost=0.001, noise_study=False, verbose=False):
    if not(1 <= n_paths and 1 <= n_obs <= 6):
        raise ValueError("Use positive path count and 1-6 observation times")
    # All these synthetic-finance parameters are hard-coded in this version.
    # Physical drift mu controls SIMULATED stock paths and PRIOR filtering.
    # Risk-free r is used for Black--Scholes PRICE and cash growth.
    # S0=K=1 is normalized notional; interpret dollar figures after scaling.
    x0=1.0; strike=1.0; mu=.08; r=.03; T=.8; sigma=SIGMA
    n_steps=80*(n_obs+1)//4
    # Align observation dates exactly with a uniform simulated grid.
    if n_steps < (n_obs+1)*8:
        n_steps=(n_obs+1)*10
    if n_steps%(n_obs+1):
        n_steps=(n_steps//(n_obs+1)+1)*(n_obs+1)
    rng=np.random.default_rng(seed)
    grid, paths = simulate_exact_gbm(n_paths,x0,mu,sigma,T,n_steps,rng)
    obs_idx=np.arange(1,n_obs+1)*n_steps//(n_obs+1)
    obs_times=grid[obs_idx]
    dates=np.concatenate([[0.],obs_times,[T]])
    records=[]
    posterior_rows=[]
    first_plot=None
    names=["Noisy BS", "PF plug-in", "PF averaged", "VI plug-in",
           "VI averaged", "VI adaptive", "Oracle diagnostic"]
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)

    for p in range(n_paths):
        true_prices=paths[p]
        # SIMULATED observer receives noisy STOCK PRICES. true_prices stays hidden
        # from PF/VI but is retained for diagnostic comparison and execution.
        obs_vals = true_prices[obs_idx] + np.sqrt(R)*rng.standard_normal(n_obs)
        print(f"\nPath {p+1}/{n_paths}, observations {np.round(obs_vals,4).tolist()}", flush=True)
        pf=particle_filter_at_obs(obs_vals, obs_times, x0=x0,mu=mu,sigma=sigma,R=R,
                                  K=strike,r=r,T=T,n_particles=n_particles,seed=seed+p*101)
        vi, last_flow, vi_seconds=train_prefix_flows(
           obs_vals,obs_times,x0=x0,mu=mu,R=R,steps=epochs,
           batch=32,batch_t=2,n_samples=n_samples,K=strike,r=r,T=T,
           seed=seed+p*43,print_training=verbose)
        for k,tk in enumerate(obs_times):
            posterior_rows.append({"path":p,"t":tk,"true_S":true_prices[obs_idx[k]],
                 "noisy_price":obs_vals[k],"pf_mean":pf[k]["mean"],
                 "pf_sd":pf[k]["sd"],"vi_mean":vi[k]["mean"],
                 "vi_sd":vi[k]["sd"],"vi_loss":vi[k]["loss_last"],
                 "vi_training_s":vi[k]["training_sec"],
                 "pf_delta":pf[k]["delta_avg"], "vi_delta":vi[k]["delta_avg"]})
        h0=float(bs_delta(x0,strike,r,sigma,T))
        h_noisy=np.array([h0]+[float(bs_delta(max(1e-8,s),strike,r,sigma,T-t))
                                 for t,s in zip(obs_times,obs_vals)])
        h_pf_plugin=np.array([h0]+[o["delta_plugin"] for o in pf])
        h_pf_avg=np.array([h0]+[o["delta_avg"] for o in pf])
        h_vi_plugin=np.array([h0]+[o["delta_plugin"] for o in vi])
        h_vi_avg=np.array([h0]+[o["delta_avg"] for o in vi])
        # ORACLE obtains hedge deltas from TRUE latent prices; diagnostic only.
        h_oracle=np.array([h0]+[float(bs_delta(s,strike,r,sigma,T-t))
                                    for t,s in zip(obs_times,true_prices[obs_idx])])
        # Posterior coefficient of variation: sd(S)/E[S].
        # This makes the adaptive threshold dimensionless, like call delta.
        cv=np.array([0.]+[o["sd"]/max(o["mean"],1e-10) for o in vi])
        targets=[h_noisy,h_pf_plugin,h_pf_avg,h_vi_plugin,
                 h_vi_avg,h_vi_avg,h_oracle]
        # Execution prices are the latent simulated path for controlled tests.
        # This separates estimation error from execution-price noise but makes
        # the experiment deliberately less realistic than live trading.
        prices_at_decisions=true_prices[np.r_[0,obs_idx,n_steps]]
        first_actual_adaptive = None
        for name,h_target in zip(names,targets):
            result=hedge_one_path(prices_at_decisions, dates, strike,r,sigma,T,
                              h_target,trade_cost=cost,
                              adaptive=(name=="VI adaptive"),cv=cv)
            if name == "VI adaptive":
                first_actual_adaptive = result["positions"]
            records.append({"path":p,"method":name,"pnl":result["pnl"],
                      "cost":result["cost"],"rebalances":result["rebalances"],
                      "vi_training_total_s":vi_seconds})
        if first_plot is None:
            first_plot={"grid":grid,"path":true_prices.copy(),"obs_times":obs_times,
                        "obs":obs_vals,"pf":pf,"vi":vi,"times":dates,
                        "targets":dict(zip(names,targets)),
                        "adaptive_positions":first_actual_adaptive}

    def save_table(fn,rows):
        with open(out/fn,"w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader();w.writerows(rows)
    save_table("per_path_hedging.csv",records)
    save_table("posterior_diagnostics.csv",posterior_rows)
    summary=[]
    # Compute sample statistics across independently simulated paths for each
    # strategy, then write them to summary.csv. More paths -> more reliable RMSE.
    for name in names:
        subset=[o for o in records if o["method"]==name]
        pnl=np.array([o["pnl"] for o in subset]); cs=np.array([o["cost"] for o in subset])
        summary.append({"method":name,"n_paths":len(subset),
                        "mean_pnl":float(pnl.mean()),
                        "pnl_rmse":float(np.sqrt(np.mean(pnl**2))),
                        "cvar95_loss":cvar_loss(pnl),
                        "avg_trading_cost":float(cs.mean()),
                        "avg_rebalances":float(np.mean([o["rebalances"] for o in subset]))})
    # CSV outputs so you can compute your own statistics without retraining:
    #   per_path_hedging.csv     : individual path/method P&L and trading costs
    #   posterior_diagnostics.csv: true latent vs observed vs PF vs VI estimates
    #   summary.csv             : across-path RMSE, CVaR, costs, rebalance counts
    save_table("summary.csv",summary)

    # Figures accessible from main script as PNGs.
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(first_plot["grid"],first_plot["path"],label="True latent stock price",lw=2)
    ax.plot(first_plot["obs_times"],first_plot["obs"],"x",ms=8,label="Noisy observations")
    ax.plot(first_plot["obs_times"],[v["mean"] for v in first_plot["pf"]],"o--",label="Particle-filter mean")
    ax.plot(first_plot["obs_times"],[v["mean"] for v in first_plot["vi"]],"s--",label="Method 2 flow mean")
    ax.set(xlabel="Time (years)",ylabel="Normalized stock price",title="Latent GBM and causal estimates")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(out/"latent_vs_filtered.png",dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    for name in ["Noisy BS","PF averaged","VI averaged","VI adaptive","Oracle diagnostic"]:
        ax.step(first_plot["times"][:-1],first_plot["targets"][name],where="post",label=name)
    ax.step(first_plot["times"][:-1], first_plot["adaptive_positions"],
            where="post",label="VI adaptive: actual holdings",ls="--",lw=2.5)
    ax.set(xlabel="Time (years)",ylabel="Call delta / hedge shares",
           title="Hedge decisions on first synthetic path")
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/"target_deltas.png",dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    plotnames=["Noisy BS","PF averaged","VI plug-in","VI averaged","VI adaptive"]
    ax.bar(range(len(plotnames)),[next(s["pnl_rmse"] for s in summary if s["method"]==n)
                                  for n in plotnames])
    ax.set_xticks(range(len(plotnames)),plotnames,rotation=17,ha="right")
    ax.set(ylabel="RMSE of terminal hedging P&L",title=f"Synthetic hedging (n={n_paths}; descriptive only for small n)")
    fig.tight_layout();fig.savefig(out/"hedge_rmse.png",dpi=170);plt.close(fig)

    # Store actual settings for reproducibility. The three PNGs are examples
    # of inference quality, hedge targets/holdings, and hedging P&L RMSE.
    metadata={"model":"GBM, latent S; additive Gaussian observation noise in S",
        "S0":x0,"mu_physical":mu,"sigma":sigma,"r":r,"T":T,"K":strike,
        "R_price":R,"n_paths":n_paths,"n_obs":n_obs,"n_steps":n_steps,
        "train_steps_per_prefix":epochs,"posterior_samples":n_samples,
        "particle_count":n_particles,"transaction_cost_rate":cost,
        "seed":seed,"filtering":"causal observation prefixes",
        "execution_assumption":"trades executed at latent simulator price, unavailable as signal",
        "note":"Small-n summaries are illustrative; oracle uses latent prices and is NOT a deployable strategy."}
    (out/"experiment_config.json").write_text(json.dumps(metadata,indent=2))
    print("\nSUMMARY (multiply P&L by 100 for a $100 notional scaled interpretation):")
    for s in summary:
        print(f"{s['method']:20} RMSE={s['pnl_rmse']:.5f} "
              f"CVaR95={s['cvar95_loss']:.5f} "
              f"cost={s['avg_trading_cost']:.5f} trades={s['avg_rebalances']:.2f}")
    print("\nSaved results to",out.resolve())
    return summary,posterior_rows,records


# Before expensive neural training, check two known analytic facts:
# (a) the standard B-S European call price and delta at S=K=100;
# (b) the self-financing cash-account calculation for a single hedge held to T.
# These tests ensure basic arithmetic but do not validate posterior quality.
def sanity_checks():
    """Deterministic high-value safeguards on pricing and cash accounting."""
    call=bs_call(100.,100.,.05,.2,1.)
    delta=bs_delta(100.,100.,.05,.2,1.)
    assert abs(call-10.450583572185565)<1e-6,call
    assert abs(delta-.6368306511756191)<1e-6,delta
    # An exact simulation with known S0 and no intermediate rebalance.
    S0=1.;ST=1.1; K=1.;r=0.03;sig=.1;T=.8
    h=float(bs_delta(S0,K,r,sig,T))
    res=hedge_one_path([S0,ST],[0.,T],K,r,sig,T,[h],trade_cost=0)
    manual=(bs_call(S0,K,r,sig,T)-h*S0)*math.exp(r*T)+h*ST-max(ST-K,0.)
    assert np.isclose(res["pnl"],manual)
    print("Pricing and hedging accounting sanity checks passed.")


# CLI ENTRY POINT. Defaults intentionally perform only a SMALL SMOKE TEST:
#   --paths: number of independent synthetic GBM paths
#   --obs: observations/rebalancing decision dates per path (max 6)
#   --train-steps: Adam steps for EACH observation-prefix flow
#   --samples: posterior samples used to estimate delta/moments
#   --particles: particles in independent particle-filter benchmark
#   --R: additive PRICE observation noise variance (standard dev=sqrt(R))
#   --cost: one-way proportional transaction cost e.g. .001 = 0.1%
#   --seed: reproducible RNG seed
#   --outdir: folder containing CSVs, figures, and experiment_config.json
#   --verbose-train: show intermediate loss lines during NN optimization
# EXAMPLE:
#   python GBM_Variational_Hedging_Explained.py --paths 8 --obs 3 \
#     --train-steps 200 --samples 1024 --particles 12000 --outdir pilot
def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths",type=int,default=1)
    p.add_argument("--obs",type=int,default=2)
    p.add_argument("--train-steps",type=int,default=40)
    p.add_argument("--samples",type=int,default=512)
    p.add_argument("--particles",type=int,default=3000)
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--R",type=float,default=.0009,help="Price-observation noise variance")
    p.add_argument("--cost",type=float,default=.001,help="One-way transaction cost fraction")
    p.add_argument("--outdir",default="results_local")
    p.add_argument("--verbose-train",action="store_true")
    a=p.parse_args()
    sanity_checks()
    run_experiment(n_paths=a.paths,n_obs=a.obs,epochs=a.train_steps,n_samples=a.samples,
                   n_particles=a.particles,seed=a.seed,outdir=a.outdir,R=a.R,
                   cost=a.cost,verbose=a.verbose_train)

if __name__=="__main__":
    main()
