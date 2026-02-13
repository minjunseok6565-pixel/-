from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Union, Optional, List

Number = Union[int, float]

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

@dataclass
class MeanAccumulator:
    n: int = 0
    # nested sums
    sums: Dict[str, Any] = field(default_factory=dict)

    def add(self, x: Mapping[str, Any]) -> None:
        self.n += 1
        _acc_dict(self.sums, x)

    def mean(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {}
        return _mean_dict(self.sums, self.n)

def _acc_dict(acc: Dict[str, Any], x: Mapping[str, Any]) -> None:
    for k, v in x.items():
        if v is None:
            continue
        if _is_num(v):
            acc[k] = float(acc.get(k, 0.0)) + float(v)
        elif isinstance(v, Mapping):
            sub = acc.get(k)
            if not isinstance(sub, dict):
                sub = {}
                acc[k] = sub
            _acc_dict(sub, v)
        # ignore lists/strings/etc. (calibration averages focus on numeric aggregates)

def _mean_dict(sums: Dict[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sums.items():
        if _is_num(v):
            out[k] = float(v) / float(n)
        elif isinstance(v, dict):
            out[k] = _mean_dict(v, n)
    return out


def _quantile_sorted(sorted_vals: List[float], q: float) -> float:
    """Compute a quantile from a pre-sorted list using linear interpolation."""
    if not sorted_vals:
        return 0.0
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])

    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac

def _acc_dict_stats(
    sums: Dict[str, Any],
    sumsqs: Dict[str, Any],
    values: Dict[str, Any],
    x: Mapping[str, Any],
) -> None:
    for k, v in x.items():
        if v is None:
            continue
        if _is_num(v):
            fv = float(v)
            sums[k] = float(sums.get(k, 0.0)) + fv
            sumsqs[k] = float(sumsqs.get(k, 0.0)) + (fv * fv)
            lst = values.get(k)
            if not isinstance(lst, list):
                lst = []
                values[k] = lst
            lst.append(fv)
        elif isinstance(v, Mapping):
            sub_s = sums.get(k)
            if not isinstance(sub_s, dict):
                sub_s = {}
                sums[k] = sub_s
            sub_ss = sumsqs.get(k)
            if not isinstance(sub_ss, dict):
                sub_ss = {}
                sumsqs[k] = sub_ss
            sub_v = values.get(k)
            if not isinstance(sub_v, dict):
                sub_v = {}
                values[k] = sub_v
            _acc_dict_stats(sub_s, sub_ss, sub_v, v)
        # ignore lists/strings/etc.

def _std_dict(sums: Dict[str, Any], sumsqs: Dict[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if n <= 1:
        for k, v in sums.items():
            if _is_num(v):
                out[k] = 0.0
            elif isinstance(v, dict):
                out[k] = _std_dict(
                    v,
                    sumsqs.get(k, {}) if isinstance(sumsqs.get(k), dict) else {},
                    n,
                )
        return out

    for k, v in sums.items():
        if _is_num(v):
            s = float(v)
            ss = float(sumsqs.get(k, 0.0))
            mean = s / float(n)
            var = (ss / float(n)) - (mean * mean)  # population variance
            if var < 0.0:  # numerical guard
                var = 0.0
            out[k] = math.sqrt(var)
        elif isinstance(v, dict):
            sub_ss = sumsqs.get(k)
            out[k] = _std_dict(v, sub_ss if isinstance(sub_ss, dict) else {}, n)
    return out

def _sorted_values_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy `values` but sort every numeric-leaf list once (for cheap multi-quantiles)."""
    out: Dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(v, list):
            out[k] = sorted(float(x) for x in v)
        elif isinstance(v, dict):
            out[k] = _sorted_values_dict(v)
    return out

def _pct_dict(values_sorted: Dict[str, Any], q: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in values_sorted.items():
        if isinstance(v, list):
            out[k] = _quantile_sorted(v, q)
        elif isinstance(v, dict):
            out[k] = _pct_dict(v, q)
    return out

@dataclass
class StatsAccumulator:
    """Accumulates per-key mean/std/percentiles for nested dict metrics.

    - mean/std use global sample count `n` (same semantics as MeanAccumulator.mean()).
    - std is population std (divide by n).
    - percentiles are computed from observed per-sample values.
    """
    n: int = 0
    sums: Dict[str, Any] = field(default_factory=dict)
    sumsqs: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)

    def add(self, x: Mapping[str, Any]) -> None:
        self.n += 1
        _acc_dict_stats(self.sums, self.sumsqs, self.values, x)

    def mean(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {}
        return _mean_dict(self.sums, self.n)

    def std(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {}
        return _std_dict(self.sums, self.sumsqs, self.n)

    def percentiles(self, *, pcts: Optional[List[int]] = None) -> Dict[str, Any]:
        if self.n <= 0:
            return {}
        if not pcts:
            pcts = [10, 50, 90]
        values_sorted = _sorted_values_dict(self.values)
        out: Dict[str, Any] = {}
        for p in pcts:
            q = float(p) / 100.0
            out[f"p{int(p)}"] = _pct_dict(values_sorted, q)
        return out

def safe_div(a: float, b: float) -> float:
    return (float(a) / float(b)) if b else 0.0

def pct(made: float, att: float) -> float:
    return safe_div(made, att) * 100.0
