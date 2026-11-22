"""Frozen eligible-wallet bootstrap functions from the historical six-view analysis."""
import numpy as np
BOOTSTRAPS=1999

def normalize(x):
    x = np.asarray(x, float)
    return x / x.sum() if x.sum() else np.zeros_like(x)


def jsd(x, y):
    p, q = normalize(x), normalize(y)
    m = .5 * (p + q)
    value = 0.0
    for z in (p, q):
        mask = z > 0
        value += .5 * float(np.sum(z[mask] * np.log2(z[mask] / m[mask])))
    return value


def upper(W):
    return np.asarray(W, float)[np.triu_indices(len(W), 1)]


def coallocation(X, mode):
    X = np.asarray(X, float)
    X = X[X.sum(1) > 0]
    if not len(X):
        return np.zeros((X.shape[1], X.shape[1]), float)
    totals = X.sum(1)
    P = X / totals[:, None]
    if mode == "amount":
        W = P.T @ (P * totals[:, None]) / totals.sum()
    elif mode == "equal_wallet":
        W = P.T @ P / len(P)
    elif mode == "binary":
        B = (X > 0).astype(float)
        W = B.T @ B / len(B)
    else:
        raise ValueError(mode)
    np.fill_diagonal(W, 0)
    return W


def profiles(X):
    X = np.asarray(X, float)
    X = X[X.sum(1) > 0]
    if not len(X):
        z = np.zeros(X.shape[1])
        return {"amount": z, "equal_wallet": z, "prevalence": z}
    P = X / X.sum(1)[:, None]
    prevalence = (X > 0).mean(0)
    return {
        "amount": normalize(X.sum(0)),
        "equal_wallet": P.mean(0),
        "prevalence": normalize(prevalence),
    }


def bootstrap_delta(data, rng):
    names = list(data)
    metrics = [
        "profile_amount", "profile_equal_wallet", "profile_prevalence",
        "edge_amount", "edge_equal_wallet", "edge_binary",
    ]
    draws = {name: np.empty(BOOTSTRAPS, float) for name in metrics}
    for b in range(BOOTSTRAPS):
        sampled = {}
        for name in names:
            X = data[name]
            sampled[name] = X[rng.integers(0, len(X), len(X))]
        pro = {name: profiles(X) for name, X in sampled.items()}
        nets = {
            mode: {name: coallocation(X, mode) for name, X in sampled.items()}
            for mode in ("amount", "equal_wallet", "binary")
        }
        for mode in ("amount", "equal_wallet", "prevalence"):
            bottom = jsd(pro["Harris Bottom 5%"][mode], pro["Trump Bottom 5%"][mode])
            top = jsd(pro["Harris Top 5%"][mode], pro["Trump Top 5%"][mode])
            draws[f"profile_{mode}"][b] = bottom - top
        for mode in ("amount", "equal_wallet", "binary"):
            bottom = jsd(upper(nets[mode]["Harris Bottom 5%"]), upper(nets[mode]["Trump Bottom 5%"] ))
            top = jsd(upper(nets[mode]["Harris Top 5%"]), upper(nets[mode]["Trump Top 5%"] ))
            draws[f"edge_{mode}"][b] = bottom - top
    return {
        name: {
            "estimate_bottom_minus_top": float(np.median(values)),
            "bootstrap_95_interval": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "fraction_positive": float(np.mean(values > 0)),
        }
        for name, values in draws.items()
    }
