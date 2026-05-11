import argparse
import json
import math
import numpy as np
import torch
from sgno import build_sgno_from_config

def _load_npz(path: str):
    d = np.load(path)
    keys = set(d.files)
    if 'test' in keys:
        return d['test']
    if 'u' in keys:
        return d['u']
    raise ValueError('NPZ must contain test or u')

def _nrmse(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12):
    num = torch.linalg.norm(a - b)
    den = torch.linalg.norm(b) + eps
    return (num / den).item()

def _gmean(errs: list[float], cap: float = 0.0, eps: float = 1e-12):
    if not errs:
        return math.nan
    x = np.asarray(errs, dtype=np.float64)
    if cap and cap > 0:
        x = np.minimum(x, float(cap))
    return float(np.exp(np.mean(np.log(x + float(eps)))))

def _stable_step(errs: list[float], tau: float, horizon: int):
    thr = float(tau)
    for i, e in enumerate(errs, start=1):
        if not math.isfinite(e):
            return int(i)
        if e > thr:
            return int(i)
    return int(horizon)

def _load_checkpoint(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--cap', type=float, default=0.0)
    ap.add_argument('--max_traj', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    spatial_dim = int(cfg['spatial_dim'])
    num_channels = int(cfg['num_channels'])
    num_points = int(cfg.get('num_points', 0))
    network_config = str(cfg['network_config'])

    ck = _load_checkpoint(args.ckpt)
    model = build_sgno_from_config(network_config, spatial_dim, num_points, num_channels).to(args.device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()

    u = _load_npz(args.data).astype(np.float32, copy=False)
    u = torch.from_numpy(u)

    initial_step = 1
    for part in network_config.split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            if k.strip().lower() == 'initial_step':
                initial_step = int(float(v.strip()))
                break

    n_traj = int(u.shape[0])
    if args.max_traj and args.max_traj > 0:
        n_traj = min(n_traj, int(args.max_traj))

    t_total = int(u.shape[1])
    horizon = min(int(args.steps), t_total - initial_step)
    horizon = max(0, horizon)

    per_traj = []
    all_errs = []
    step_errs = [[] for _ in range(horizon)]

    with torch.no_grad():
        for i in range(n_traj):
            hist = u[i, :initial_step].unsqueeze(0).to(args.device)
            errs = []
            for t in range(horizon):
                y = model(hist)
                gt = u[i, initial_step + t].to(args.device)
                e = _nrmse(y, gt)
                errs.append(float(e))
                step_errs[t].append(float(e))
                if initial_step == 1:
                    hist = y.unsqueeze(1)
                else:
                    hist = torch.cat([hist[:, 1:], y.unsqueeze(1)], dim=1)
            g = _gmean(errs[: min(100, len(errs))], cap=args.cap)
            s = _stable_step(errs, tau=args.tau, horizon=horizon)
            fin = float(errs[-1]) if errs else math.nan
            per_traj.append({'gmean100': g, 'stable_step': s, 'final_nrmse': fin})
            all_errs.extend(errs)

    mean_nrmse_by_step = [
        float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else math.nan
        for xs in step_errs
    ]
    gmean_steps = mean_nrmse_by_step[: min(100, len(mean_nrmse_by_step))]
    paper_gmean100 = _gmean(gmean_steps, cap=args.cap)
    paper_stable_step = _stable_step(mean_nrmse_by_step, tau=args.tau, horizon=horizon)

    gmeans = np.asarray([p['gmean100'] for p in per_traj], dtype=np.float64)
    steps_arr = np.asarray([p['stable_step'] for p in per_traj], dtype=np.float64)

    out = {
        'n_traj': int(n_traj),
        'horizon': int(horizon),
        'tau': float(args.tau),
        'cap': float(args.cap),
        'gmean100': float(paper_gmean100),
        'stable_step': int(paper_stable_step),
        'mean_nrmse_by_step': mean_nrmse_by_step,
        'mean_nrmse': float(np.mean(all_errs) if all_errs else math.nan),
        'median_nrmse': float(np.median(all_errs) if all_errs else math.nan),
        'median_traj_gmean100': float(np.median(gmeans) if per_traj else math.nan),
        'mean_traj_gmean100': float(np.mean(gmeans) if per_traj else math.nan),
        'median_traj_stable_step': float(np.median(steps_arr) if per_traj else math.nan),
        'q25_traj_stable_step': float(np.quantile(steps_arr, 0.25) if per_traj else math.nan),
        'q75_traj_stable_step': float(np.quantile(steps_arr, 0.75) if per_traj else math.nan),
        'per_traj': per_traj,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f)

    print(args.out)

if __name__ == '__main__':
    main()
