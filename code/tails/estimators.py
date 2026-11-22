"""Frozen continuous CSN fit and bootstrap kernels; exact extraction from the analysis."""
import math
import numpy as np
from numba import njit
_WORKER={}

def clean(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return np.sort(x[np.isfinite(x) & (x > 0)])


def candidate_indices(x: np.ndarray, minimum: int, budget: int, exhaustive_limit: int) -> np.ndarray:
    n = len(x)
    if n <= exhaustive_limit:
        idx = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
        return idx[idx <= n - minimum].astype(np.int64)
    sizes = np.unique(
        np.r_[np.linspace(minimum, n, budget).astype(np.int64),
              np.geomspace(minimum, n, budget).astype(np.int64)]
    )
    idx = np.sort(n - sizes)
    idx = idx[(idx >= 0) & (idx <= n - minimum)]
    while len(idx) and np.any((idx > 0) & (x[idx - 1] == x[idx])):
        idx = np.asarray([i - np.searchsorted(x[:i][::-1] != x[i], True)
                          if i > 0 and x[i - 1] == x[i] else i for i in idx], dtype=np.int64)
        break
    return np.unique(idx)


@njit(cache=False)
def _choose_standard(logx, indices, minimum):
    n = len(logx)
    suffix = np.zeros(n + 1)
    for j in range(n - 1, -1, -1):
        suffix[j] = suffix[j + 1] + logx[j]
    best = 2.0
    best_i = -1
    best_alpha = np.nan
    for raw_i in indices:
        i = raw_i
        while i > 0 and logx[i - 1] == logx[i]:
            i -= 1
        m = n - i
        if m < minimum:
            continue
        den = suffix[i] - m * logx[i]
        if den <= 1e-12:
            continue
        alpha = 1.0 + m / den
        ks = 0.0
        for j in range(i, n):
            cdf = -math.expm1((1.0 - alpha) * (logx[j] - logx[i]))
            d1 = abs(cdf - (j - i) / m)
            d2 = abs((j - i + 1) / m - cdf)
            if d1 > ks:
                ks = d1
            if d2 > ks:
                ks = d2
            if ks > best:
                break
        if ks < best:
            best = ks
            best_i = i
            best_alpha = alpha
    return best_i, best_alpha, best


def continuous_fit(values, minimum=100, budget=500, exhaustive_limit=2000,
                   fixed=None, selector="ks") -> dict | None:
    x = clean(values)
    n = len(x)
    if n < minimum:
        return None
    if fixed is not None:
        tail = x[x >= fixed]
        if len(tail) < minimum:
            return None
        den = np.log(tail / fixed).sum()
        if den <= 1e-12:
            return None
        alpha = 1.0 + len(tail) / den
        cdf = -np.expm1((1.0 - alpha) * np.log(tail / fixed))
        before = np.arange(len(tail)) / len(tail)
        after = before + 1.0 / len(tail)
        ks = max(np.max(np.abs(cdf - before)), np.max(np.abs(after - cdf)))
        return dict(alpha=float(alpha), xmin=float(fixed), ks=float(ks), n=n,
                    n_tail=len(tail), index=n - len(tail), selector="fixed",
                    candidate_count=1, exhaustive=True)
    idx = candidate_indices(x, minimum, budget, exhaustive_limit)
    if selector == "ks":
        i, alpha, distance = _choose_standard(np.log(x), idx, minimum)
    else:
        logx = np.log(x)
        suffix = np.r_[np.cumsum(logx[::-1])[::-1], 0.0]
        best = None
        for raw_i in idx:
            i = int(raw_i)
            while i > 0 and x[i - 1] == x[i]:
                i -= 1
            m = n - i
            if m < minimum:
                continue
            den = suffix[i] - m * logx[i]
            if den <= 1e-12:
                continue
            a = 1.0 + m / den
            cdf = -np.expm1((1.0 - a) * (logx[i:] - logx[i]))
            before = np.arange(m) / m
            after = before + 1.0 / m
            if selector == "reweighted_ks":
                denom = np.sqrt(np.clip(cdf * (1.0 - cdf), 1.0 / (4 * m * m), None))
                d = max(np.max(np.abs(cdf - before) / denom),
                        np.max(np.abs(after - cdf) / denom))
            elif selector == "kuiper":
                d = np.max(after - cdf) + np.max(cdf - before)
            else:
                raise ValueError(selector)
            if best is None or d < best[0]:
                best = (float(d), i, float(a))
        if best is None:
            return None
        distance, i, alpha = best
    if i < 0:
        return None
    return dict(alpha=float(alpha), xmin=float(x[i]), ks=float(distance), n=n,
                n_tail=n - int(i), index=int(i), selector=selector,
                candidate_count=int(len(idx)), exhaustive=bool(n <= exhaustive_limit))


def _init_worker(x, observed, settings):
    _WORKER.update(x=x, observed=observed, settings=settings)
    _choose_standard(np.log(np.arange(1.0, 301.0)), np.array([0, 50, 100]), 100)


def _one_continuous(rep: int, kind: str) -> dict:
    x = _WORKER["x"]
    o = _WORKER["observed"]
    s = _WORKER["settings"]
    code = 0 if kind == "gof" else 1
    rng = np.random.default_rng(np.random.SeedSequence([s["seed"], code, rep]))
    if kind == "gof":
        body = x[x < o["xmin"]]
        nt = int(rng.binomial(len(x), o["n_tail"] / len(x))) if len(body) else len(x)
        tail = o["xmin"] * np.exp(rng.exponential(1.0 / (o["alpha"] - 1.0), nt))
        sample = np.r_[rng.choice(body, len(x) - nt, replace=True), tail] if len(body) else tail
    else:
        sample = rng.choice(x, len(x), replace=True)
    fit = continuous_fit(sample, s["minimum"], s["candidate_budget"],
                         s["exhaustive_limit"], s.get("fixed"), "ks")
    if fit is None:
        return dict(rep=rep, kind=kind, valid=False)
    return dict(rep=rep, kind=kind, valid=True, alpha=fit["alpha"], xmin=fit["xmin"],
                ks=fit["ks"], n_tail=fit["n_tail"], exceed_ge=bool(fit["ks"] >= o["ks"]),
                exceed_gt=bool(fit["ks"] > o["ks"]))
