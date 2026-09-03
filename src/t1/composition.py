from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from t1.clusters import (
    BIRTH_CLUSTERS,
    CLUSTER_ORDER,
    DISSECT_CLUSTER,
    labels_to_clusters,
)


class CompositionModel:
    """Log-linear cluster proportions with dissection decay and birth growth."""

    def __init__(
        self,
        t_85: float = 8.5,
        t_95: float = 9.5,
        rho: float = 1.2,
        pi_max: float = 0.25,
        eps: float = 1e-6,
    ):
        self.t_85 = t_85
        self.t_95 = t_95
        self.rho = rho
        self.pi_max = pi_max
        self.eps = eps
        self.pi_85: dict[str, float] = {}
        self.pi_95: dict[str, float] = {}
        self.log_slope: dict[str, float] = {}
        self.counts_85: dict[str, int] = {}
        self.counts_95: dict[str, int] = {}

    def fit(self, labels_85, labels_95) -> "CompositionModel":
        c85 = labels_to_clusters(labels_85)
        c95 = labels_to_clusters(labels_95)
        self.counts_85 = dict(Counter(c85))
        self.counts_95 = dict(Counter(c95))
        n85, n95 = len(c85), len(c95)
        for k in CLUSTER_ORDER:
            self.pi_85[k] = self.counts_85.get(k, 0) / n85
            self.pi_95[k] = self.counts_95.get(k, 0) / n95
            p0 = max(self.pi_85[k], self.eps)
            p1 = max(self.pi_95[k], self.eps)
            self.log_slope[k] = float(np.log(p1) - np.log(p0))
        return self

    def probs(self, t: float) -> dict[str, float]:
        if t <= self.t_85:
            logp = {k: np.log(self.pi_85[k] + self.eps) for k in CLUSTER_ORDER}
            return self._softmax(logp)
        if t <= self.t_95:
            w = (t - self.t_85) / (self.t_95 - self.t_85)
            logp = {
                k: (1 - w) * np.log(self.pi_85[k] + self.eps)
                + w * np.log(self.pi_95[k] + self.eps)
                for k in CLUSTER_ORDER
            }
            pi = self._softmax(logp)
            if t >= self.t_95 - 1e-9:
                pi[DISSECT_CLUSTER] = 0.0
                pi = self._renorm(pi)
            return pi

        # Extrapolation: freeze E9.5 mix, grow birth by ρ, drop dissection.
        dt = t - self.t_95
        pi = {k: self.pi_95[k] for k in CLUSTER_ORDER}
        pi[DISSECT_CLUSTER] = 0.0
        for k in BIRTH_CLUSTERS:
            pi[k] = min(self.pi_95[k] * (self.rho ** dt), self.pi_max)
        return self._renorm(pi)

    def _softmax(self, logp: dict[str, float]) -> dict[str, float]:
        m = max(logp.values())
        unnorm = {k: float(np.exp(v - m)) for k, v in logp.items()}
        return self._renorm(unnorm)

    @staticmethod
    def _renorm(pi: dict[str, float]) -> dict[str, float]:
        total = sum(pi.values())
        if total <= 0:
            raise RuntimeError("Composition probabilities summed to 0")
        return {k: v / total for k, v in pi.items()}

    def allocate(self, t: float, n: int, rng: np.random.Generator) -> dict[str, int]:
        pi = self.probs(t)
        keys = [k for k, p in pi.items() if p > 0]
        p = np.array([pi[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        counts = rng.multinomial(n, p)
        return {k: int(c) for k, c in zip(keys, counts) if c > 0}

    def to_dict(self) -> dict:
        return {
            "t_85": self.t_85,
            "t_95": self.t_95,
            "rho": self.rho,
            "pi_max": self.pi_max,
            "eps": self.eps,
            "pi_85": self.pi_85,
            "pi_95": self.pi_95,
            "log_slope": self.log_slope,
            "counts_85": self.counts_85,
            "counts_95": self.counts_95,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CompositionModel":
        payload = json.loads(Path(path).read_text())
        obj = cls(
            t_85=payload["t_85"],
            t_95=payload["t_95"],
            rho=payload["rho"],
            pi_max=payload["pi_max"],
            eps=payload["eps"],
        )
        obj.pi_85 = payload["pi_85"]
        obj.pi_95 = payload["pi_95"]
        obj.log_slope = payload["log_slope"]
        obj.counts_85 = {k: int(v) for k, v in payload["counts_85"].items()}
        obj.counts_95 = {k: int(v) for k, v in payload["counts_95"].items()}
        return obj
