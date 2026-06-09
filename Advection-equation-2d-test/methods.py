import math
import time
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class FeatureMap2D(nn.Module):
    def __init__(self, Y, T, n_freq_y=8, n_freq_t=8):
        super().__init__()
        self.Y = Y
        self.T = T
        ky = torch.arange(1, n_freq_y + 1, dtype=torch.float32).view(1, -1)
        kt = torch.arange(1, n_freq_t + 1, dtype=torch.float32).view(1, -1)
        self.register_buffer("ky", ky)
        self.register_buffer("kt", kt)

    def forward(self, y, t):
        y_scaled = 2.0 * y / self.Y - 1.0
        t_scaled = 2.0 * t / self.T - 1.0
        zy = math.pi * y_scaled * self.ky
        zt = math.pi * t_scaled * self.kt
        return torch.cat([
            y_scaled, t_scaled,
            torch.sin(zy), torch.cos(zy),
            torch.sin(zt), torch.cos(zt),
        ], dim=1)


class FourierFeatureMap2D(nn.Module):
    """
    Random Fourier Features (Tancik et al. 2020).
    Maps (y, t) -> [cos(2π·B·x), sin(2π·B·x)] where B ~ N(0, sigma²).
    Frequencies are fixed after construction (not learned).

    With n_features=11, output dim = 22 — same as FeatureMap2D with n_freq=5,
    so the same network architectures work for both feature maps.
    """

    def __init__(self, Y, T, n_features=11, sigma=1.0):
        super().__init__()
        self.Y = Y
        self.T = T
        B = torch.randn(2, n_features) * sigma
        self.register_buffer("B", B)

    def forward(self, y, t):
        y_norm = y / self.Y
        t_norm = t / self.T
        x    = torch.cat([y_norm, t_norm], dim=1)   # (N, 2)
        proj = 2.0 * math.pi * (x @ self.B)         # (N, n_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)


# ---------------------------------------------------------------------------
# Method 0a — Vanilla PINN (baseline, no gating)
# ---------------------------------------------------------------------------

class VanillaPINN(nn.Module):
    """
    Single-network PINN. Wraps a plain nn.Sequential so it shares the same
    (gates, expert_vals, u) interface as MoEModel.

    Returns trivial gates (ones) so the rest of the training code is unchanged;
    the load-balancing loss weight should be set to 0.0 in the config when
    using this model.
    """

    def __init__(self, feature_map: FeatureMap2D, network: nn.Module):
        super().__init__()
        self.feature_map = feature_map
        self.network = network

    def forward(self, y, t):
        feats = self.feature_map(y, t)
        u     = self.network(feats)
        N     = u.shape[0]
        gates = torch.ones(N, 1, device=u.device, dtype=u.dtype)
        return gates, u.unsqueeze(1), u


# ---------------------------------------------------------------------------
# Method 0b — MoE PINN (configurable gating)
# ---------------------------------------------------------------------------

class MoEModel(nn.Module):
    """
    Mixture-of-Experts PINN.

    gating="continuous" — softmax gates; each expert contributes a continuous
                          fraction to every point (e.g. 0.3 and 0.7).
    gating="binary"     — winner-takes-all routing via straight-through estimator:
                          forward assigns each point to exactly one expert (one-hot
                          argmax), while backward passes gradients through the softmax
                          so the gating network can still learn.

    Controlled by config keys: use_moe=True, moe_gating="continuous"|"binary"
    """

    def __init__(
        self,
        feature_map: FeatureMap2D,
        gating_net: nn.Module,
        experts: list,
        gating: str = "continuous",
        k: int = 2,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.gating_net  = gating_net
        self.experts     = nn.ModuleList(experts)
        self.gating      = gating
        self.k           = k   # used only when gating="topk"

        if gating not in ("continuous", "binary", "topk"):
            raise ValueError(f"gating must be 'continuous', 'binary', or 'topk', got '{gating}'")

    def forward(self, y, t):
        feats  = self.feature_map(y, t)
        logits = self.gating_net(feats)
        soft   = torch.softmax(logits, dim=1)

        if self.gating == "binary":
            # Straight-through: one-hot forward, softmax gradient backward
            hard  = torch.zeros_like(soft)
            hard.scatter_(1, soft.argmax(dim=1, keepdim=True), 1.0)
            gates = hard - soft.detach() + soft

        elif self.gating == "topk":
            # Top-k sparse gating: only the k largest experts contribute,
            # their weights renormalised to sum to 1. Gradients flow through
            # the softmax values of the selected experts.
            k = min(self.k, soft.shape[1])
            _, topk_idx  = soft.topk(k, dim=1)
            mask         = torch.zeros_like(soft)
            mask.scatter_(1, topk_idx, 1.0)
            masked       = soft * mask
            gates        = masked / masked.sum(dim=1, keepdim=True)

        else:  # continuous
            gates = soft

        expert_outs = torch.stack(
            [e(feats).squeeze(-1) for e in self.experts], dim=1
        )
        u = torch.sum(gates * expert_outs, dim=1, keepdim=True)
        return gates, expert_outs, u


# ---------------------------------------------------------------------------
# PDE residual and loss computation
# ---------------------------------------------------------------------------

def pde_residual(model, y, t, c, v):
    """Returns (u, u_y, u_t, u_yy, residual, gates, expert_vals, ini_pred)."""
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    gates, expert_vals, u = model(y, t)

    u_t  = torch.autograd.grad(u, t,   grad_outputs=torch.ones_like(u),   create_graph=True)[0]
    u_y  = torch.autograd.grad(u, y,   grad_outputs=torch.ones_like(u),   create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    r = u_t + c * u_y - v * u_yy

    zero = torch.zeros_like(t)
    _, _, ini_pred = model(y, zero)

    return u, u_y, u_t, u_yy, r, gates, expert_vals, ini_pred


def compute_losses(model, y_batch, t_batch, c, v, K, ic_fn=torch.sin):
    """
    Returns unweighted individual losses and gates.

    ic_fn : callable applied to y giving the initial condition u(y, 0).
    K     : load-balancing scale factor (K=0 disables the load loss).
    """
    _, _, _, _, r, gates, _, ini_pred = pde_residual(model, y_batch, t_batch, c, v)

    loss_pde  = torch.mean(r ** 2)
    loss_ini  = torch.mean((ic_fn(y_batch) - ini_pred) ** 2)
    gbar      = torch.mean(gates, dim=0)
    loss_load = K * torch.sum(gbar ** 2)

    return {"pde": loss_pde, "ini": loss_ini, "load": loss_load}, gates


# ---------------------------------------------------------------------------
# Method 1 — SoftAdapt (adaptive loss weighting)
# ---------------------------------------------------------------------------

class SoftAdapt:
    """
    Dynamically reweights loss terms so that slower-improving terms receive
    larger weights.

    Controlled by config key: use_softadapt
    """

    def __init__(self, loss_names, base_weights=None, beta=5.0, eps=1e-8, enabled=True):
        self.loss_names  = loss_names
        self.beta        = beta
        self.eps         = eps
        self.enabled     = enabled
        self.prev_losses = None

        if base_weights is None:
            base_weights = {name: 1.0 for name in loss_names}
        self.base_weights    = base_weights
        self.current_weights = {name: float(base_weights[name]) for name in loss_names}

    def __call__(self, losses, update=True):
        def _base_tensors():
            return {
                name: torch.tensor(
                    self.base_weights[name],
                    device=losses[name].device,
                    dtype=losses[name].dtype,
                )
                for name in self.loss_names
            }

        if not self.enabled:
            return _base_tensors()

        if self.prev_losses is None:
            self.prev_losses = {name: losses[name].detach().clone() for name in self.loss_names}
            return _base_tensors()

        if not update:
            return {
                name: torch.tensor(
                    self.current_weights[name],
                    device=losses[name].device,
                    dtype=losses[name].dtype,
                )
                for name in self.loss_names
            }

        ratios = torch.stack([
            losses[name].detach() / (self.prev_losses[name] + self.eps)
            for name in self.loss_names
        ])
        ratios -= torch.max(ratios)
        adaptive_factors = len(self.loss_names) * torch.softmax(self.beta * ratios, dim=0)

        weights = {
            name: self.base_weights[name] * adaptive_factors[i]
            for i, name in enumerate(self.loss_names)
        }
        self.current_weights = {name: float(weights[name].detach().cpu()) for name in self.loss_names}
        self.prev_losses     = {name: losses[name].detach().clone() for name in self.loss_names}
        return weights


# ---------------------------------------------------------------------------
# Method 2 — Adaptive collocation refinement
# ---------------------------------------------------------------------------

@torch.no_grad()
def adaptive_refine(model, yt_old, Y, T, c, v,
                    n_candidates=8000, replace_fraction=0.2, gamma=50.0):
    """
    Replaces a fraction of collocation points with high-residual candidates.

    Controlled by config key: use_adaptive_refine
    """
    dev       = yt_old.device
    n_old     = yt_old.shape[0]
    n_replace = max(1, int(replace_fraction * n_old))
    n_keep    = n_old - n_replace

    y_cand = Y * torch.rand(n_candidates, 1, device=dev)
    t_cand = T * torch.rand(n_candidates, 1, device=dev)

    with torch.enable_grad():
        _, _, _, _, r, _, _, _ = pde_residual(model, y_cand, t_cand, c, v)
    r2 = r.detach().squeeze() ** 2

    med = torch.median(r2)
    r2  = torch.clamp(r2, min=med.item(), max=(gamma * med).item()) if med.item() > 0 else r2 + 1e-12

    probs   = r2 / r2.sum()
    idx_new = torch.multinomial(probs, num_samples=n_replace, replacement=False)
    yt_hard = torch.cat([y_cand[idx_new], t_cand[idx_new]], dim=1).detach()

    perm    = torch.randperm(n_old, device=dev)
    yt_keep = yt_old[perm[:n_keep]].detach()

    yt_new = torch.cat([yt_keep, yt_hard], dim=0)
    return yt_new[torch.randperm(yt_new.shape[0], device=dev)]


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(model, all_params, domain_params, config, eval_fn=None):
    """
    Train a PINN (vanilla or MoE) with any combination of the four methods.

    Parameters
    ----------
    model : VanillaPINN | MoEModel | callable
        Any callable with signature model(y, t) -> (gates, expert_vals, u).
        Use VanillaPINN or MoEModel for standard setups.
    all_params : list[nn.Parameter]
        All trainable parameters (passed to optimizer).
    domain_params : dict
        Y, T   -- domain extents
        c, v   -- advection speed and diffusion coefficient
        N_f    -- number of collocation points
        K      -- load-balancing scale factor (set to 0 for VanillaPINN)
    config : dict
        # Method toggles
        use_moe             : bool        -- True -> MoEModel, False -> VanillaPINN
        moe_gating          : str         -- "continuous" | "binary"  (only when use_moe=True)
        use_softadapt       : bool        -- Method 1
        use_adaptive_refine : bool        -- Method 2
        use_lbfgs           : bool        -- Method 3

        # Loss weights
        base_weights        : dict        -- {"pde": w1, "ini": w2, "load": w3}
                                             load weight is forced to 0 when use_moe=False

        # SoftAdapt params
        softadapt_beta      : float

        # Adam params
        adam_epochs         : int
        adam_lr             : float
        adam_step_size      : int
        adam_gamma          : float

        # Adaptive refinement params
        refine_every        : int
        n_candidates        : int
        replace_fraction    : float
        refine_gamma        : float

        # L-BFGS params
        lbfgs_max_iter      : int

        # Misc
        log_every           : int
        ic_fn               : callable    -- default torch.sin

    Returns
    -------
    hist : dict
        Per-epoch lists keyed by loss component name + "total".
    """
    Y   = domain_params["Y"]
    T   = domain_params["T"]
    c   = domain_params["c"]
    v   = domain_params["v"]
    N_f = domain_params["N_f"]
    K   = domain_params["K"]

    ic_fn        = config.get("ic_fn", torch.sin)
    base_weights = dict(config.get("base_weights", {"pde": 1.0, "ini": 10.0, "load": 1e-2}))

    # No load-balancing concept for a vanilla single network
    if not config.get("use_moe", True):
        base_weights["load"] = 0.0

    loss_names = list(base_weights.keys())
    dev        = all_params[0].device

    # Collocation points
    y_f = Y * torch.rand(N_f, 1, device=dev)
    t_f = T * torch.rand(N_f, 1, device=dev)

    # Method 1 — SoftAdapt
    softadapt = SoftAdapt(
        loss_names   = loss_names,
        base_weights = base_weights,
        beta         = config.get("softadapt_beta", 5.0),
        enabled      = config.get("use_softadapt", False),
    )

    # Adam + LR scheduler
    optimizer = torch.optim.Adam(all_params, lr=config.get("adam_lr", 1e-3))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = config.get("adam_step_size", 2500),
        gamma     = config.get("adam_gamma", 0.5),
    )

    hist = {name: [] for name in loss_names}
    hist["total"]         = []
    hist["wall_time"]     = []   # cumulative seconds at end of each epoch
    hist["eval_epochs"]   = []   # epochs at which eval_fn was called
    hist["eval_l2_rel"]   = []
    hist["eval_max_err"]  = []

    t_start      = time.time()
    eval_every   = config.get("eval_every", 0)   # 0 = no mid-training eval
    adam_epochs  = config.get("adam_epochs", 10000)
    refine_every = config.get("refine_every", 400)
    log_every    = config.get("log_every", 500)

    # ---- Adam phase --------------------------------------------------------
    for epoch in range(1, adam_epochs + 1):
        losses, _ = compute_losses(model, y_f, t_f, c, v, K, ic_fn)
        weights   = softadapt(losses)

        loss_total = sum(weights[name] * losses[name] for name in loss_names)

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        hist["total"].append(loss_total.item())
        for name in loss_names:
            hist[name].append(losses[name].item())
        hist["wall_time"].append(time.time() - t_start)

        if eval_fn is not None and eval_every > 0 and epoch % eval_every == 0:
            metrics = eval_fn(model, epoch)
            hist["eval_epochs"].append(epoch)
            hist["eval_l2_rel"].append(metrics.get("l2_rel", float("nan")))
            hist["eval_max_err"].append(metrics.get("max_err", float("nan")))

        # Method 2 — adaptive collocation refinement
        if config.get("use_adaptive_refine", False) and epoch % refine_every == 0:
            yt_f = adaptive_refine(
                model            = model,
                yt_old           = torch.cat([y_f, t_f], dim=1),
                Y                = Y,
                T                = T,
                c                = c,
                v                = v,
                n_candidates     = config.get("n_candidates", 8000),
                replace_fraction = config.get("replace_fraction", 0.2),
                gamma            = config.get("refine_gamma", 50.0),
            )
            y_f = yt_f[:, 0:1]
            t_f = yt_f[:, 1:2]

        if epoch % log_every == 0:
            parts = " | ".join(f"{n}={losses[n].item():.3e}" for n in loss_names)
            print(f"Adam {epoch:5d} | total={loss_total.item():.3e} | {parts}")

    # ---- Method 3 — L-BFGS phase ------------------------------------------
    if config.get("use_lbfgs", False):
        lbfgs_max_iter = config.get("lbfgs_max_iter", 600)
        lbfgs = torch.optim.LBFGS(
            all_params,
            lr             = 1.0,
            max_iter       = lbfgs_max_iter,
            max_eval       = lbfgs_max_iter,
            history_size   = 50,
            line_search_fn = "strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            losses_c, _ = compute_losses(model, y_f, t_f, c, v, K, ic_fn)
            weights_c   = softadapt(losses_c, update=False)
            loss_c      = sum(weights_c[name] * losses_c[name] for name in loss_names)
            loss_c.backward()
            return loss_c

        print("\nStarting L-BFGS...")
        final_loss = lbfgs.step(closure)
        print(f"L-BFGS final loss = {final_loss.item():.6e}")

    return hist


# ---------------------------------------------------------------------------
# Finite-difference reference solver
# ---------------------------------------------------------------------------

def fd_solve(domain_params, N_y=512, N_t_plot=1000):
    """
    Pseudo-spectral solution for u_t + c*u_y = v*u_yy on periodic [0,Y].
    IC: u(y,0) = sin(y).   Exact: u(y,t) = exp(-v*t)*sin(y - c*t).

    Because the PDE is linear with constant coefficients each Fourier mode
    evolves exactly:  û(k,t) = û(k,0) * exp(-(v*k² + i*c*k)*t).
    No time-stepping is needed — the solution is computed directly at any t,
    so there are no stability constraints or timestep size issues.

    Parameters
    ----------
    domain_params : dict   Y, T, c, v
    N_y           : int    spatial grid points (controls spatial resolution)
    N_t_plot      : int    number of time snapshots to record error history at

    Returns
    -------
    dict with keys:
        u_final, u_exact_final, y_grid, t_grid,
        l2_rel_history, max_err_history,
        l2_rel_final, max_err_final,
        solve_time_sec, N_y, N_t_plot
    """
    Y = domain_params["Y"]
    T = domain_params["T"]
    c = domain_params["c"]
    v = domain_params["v"]

    y     = np.linspace(0.0, Y, N_y, endpoint=False)
    t_all = np.linspace(0.0, T, N_t_plot + 1)
    k     = (2.0 * np.pi / Y) * np.fft.fftfreq(N_y, d=1.0 / N_y)

    u0_hat = np.fft.fft(np.sin(y))

    def spectral_solution(t_val):
        prop = np.exp(-(v * k ** 2 + 1j * c * k) * t_val)
        return np.real(np.fft.ifft(u0_hat * prop))

    def analytic(t_val):
        return np.exp(-v * t_val) * np.sin(y - c * t_val)

    def errors(u, t_val):
        ex  = analytic(t_val)
        err = u - ex
        return (float(np.linalg.norm(err) / np.linalg.norm(ex)),
                float(np.max(np.abs(err))))

    l2_history  = np.empty(N_t_plot + 1)
    max_history = np.empty(N_t_plot + 1)

    t_wall = time.time()
    for i, t_val in enumerate(t_all):
        u = spectral_solution(t_val)
        l2_history[i], max_history[i] = errors(u, t_val)
    solve_time = time.time() - t_wall

    u_final       = spectral_solution(T)
    u_exact_final = analytic(T)

    return {
        "u_final":          u_final,
        "u_exact_final":    u_exact_final,
        "y_grid":           y,
        "t_grid":           t_all,
        "l2_rel_history":   l2_history,
        "max_err_history":  max_history,
        "l2_rel_final":     float(l2_history[-1]),
        "max_err_final":    float(max_history[-1]),
        "solve_time_sec":   solve_time,
        "N_y":              N_y,
        "N_t_plot":         N_t_plot,
    }
