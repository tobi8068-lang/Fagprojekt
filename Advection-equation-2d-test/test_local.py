"""
Lightweight local smoke-test.
Samples 5 random configs from the main sweep and runs each for a few epochs
on a single seed to verify the full pipeline works end-to-end.

Usage:
    python test_local.py
    python test_local.py --n 3        # run 3 configs instead of 5
    python test_local.py --seed 42    # different random sample

Expected runtime on CPU laptop: ~2-5 minutes total.
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from configs import CONFIGS, DOMAIN
from train import build_model, evaluate, solution_grid

# ---------------------------------------------------------------------------
# Lightweight overrides — replace production settings with quick-test values
# ---------------------------------------------------------------------------

_TEST_OVERRIDES = {
    "adam_epochs":    5000,
    "adam_step_size": 1250,
    "refine_every":   50,
    "n_candidates":   2000,
    "lbfgs_max_iter": 50,
    "eval_every":     50,
    "log_every":      100,
}

_TEST_DOMAIN = {**DOMAIN, "N_f": 15000}

SEED = 1234


def make_test_cfg(cfg):
    return {**cfg, **_TEST_OVERRIDES}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=5,    help="number of configs to sample")
    parser.add_argument("--seed", type=int, default=42,   help="RNG seed for config sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    sampled = random.sample(CONFIGS, min(args.n, len(CONFIGS)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Sampled: {len(sampled)} / {len(CONFIGS)} configs  (sample seed={args.seed})")
    print("=" * 60)

    os.makedirs("results_test", exist_ok=True)
    results = []

    from methods import train as run_training

    for i, cfg in enumerate(sampled):
        test_cfg = make_test_cfg(cfg)
        print(f"\n[{i+1}/{len(sampled)}] {test_cfg['name']}")
        print("-" * 40)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model      = build_model(test_cfg, device)
        all_params = list(model.parameters())
        n_params   = sum(p.numel() for p in all_params)
        print(f"  Parameters : {n_params:,}")

        eval_fn = lambda m, e: evaluate(m, device, n_grid=100)

        t0   = time.time()
        hist = run_training(model, all_params, _TEST_DOMAIN, test_cfg, eval_fn=eval_fn)
        elapsed = time.time() - t0

        final = evaluate(model, device, n_grid=100)
        grid  = solution_grid(model, device, n_grid=100)
        print(f"  L2 rel     : {final['l2_rel']:.4e}")
        print(f"  Max err    : {final['max_err']:.4e}")
        print(f"  Wall time  : {elapsed:.2f}s")

        out_path = f"results_test/{test_cfg['name']}_seed{SEED}.npz"
        np.savez(
            out_path,
            config_name          = test_cfg["name"],
            seed                 = SEED,
            use_moe              = test_cfg["use_moe"],
            use_rff              = test_cfg.get("use_rff", False),
            use_fd_deriv         = test_cfg.get("use_fd_deriv", True),
            use_softadapt        = test_cfg["use_softadapt"],
            use_adaptive_refine  = test_cfg["use_adaptive_refine"],
            use_lbfgs            = test_cfg["use_lbfgs"],
            hist_total           = hist["total"],
            hist_pde             = hist["pde"],
            hist_ini             = hist["ini"],
            hist_load            = hist["load"],
            hist_wall_time       = hist["wall_time"],
            eval_epochs          = hist["eval_epochs"],
            eval_l2_rel          = hist["eval_l2_rel"],
            eval_max_err         = hist["eval_max_err"],
            l2_rel_final         = final["l2_rel"],
            max_err_final        = final["max_err"],
            grid_x               = grid["grid_x"],
            grid_y               = grid["grid_y"],
            grid_t_vals          = grid["grid_t_vals"],
            grid_u_pred          = grid["grid_u_pred"],
            grid_u_exact         = grid["grid_u_exact"],
            total_time_sec       = elapsed,
        )
        print(f"  Saved      : {out_path}")
        results.append((test_cfg["name"], final["l2_rel"], final["max_err"], elapsed))

    # ---- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'Config':<40}  {'L2 rel':>10}  {'Time (s)':>10}")
    print(f"{'-'*40}  {'-'*10}  {'-'*10}")
    for name, l2, _, t in results:
        print(f"{name:<40}  {l2:>10.4e}  {t:>10.2f}")
    print(f"\nResults written to results_test/")


if __name__ == "__main__":
    main()
