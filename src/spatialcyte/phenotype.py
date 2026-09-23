"""Stage 3 — phenotype.

Per-marker positivity calls (2-component GMM, Otsu fallback) plus config-driven
hierarchical gating (first match wins) → ``obs['phenotype']``. Panel-agnostic:
the gates are declared in the panel config, so any lineage panel works.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from spatialcyte.config import PanelConfig


def mark_positive(
    adata: ad.AnnData,
    markers: list[str] | None = None,
    *,
    method: str = "gmm",
    layer: str = "arcsinh",
) -> ad.AnnData:
    """Add a boolean ``obs[f'{marker}_positive']`` per marker."""
    if method not in ("gmm", "otsu"):
        raise ValueError(f"unknown positivity method {method!r} (use 'gmm' or 'otsu')")
    src = adata.layers[layer] if layer in adata.layers else adata.X
    names = list(adata.var_names)
    for m in markers or names:
        if m not in names:
            raise KeyError(f"marker {m!r} not in var_names {names}")
        col = src[:, names.index(m)]
        if hasattr(col, "toarray"):  # sparse X/layer -> dense column
            col = col.toarray()
        vals = np.asarray(col, dtype=float).ravel()
        adata.obs[f"{m}_positive"] = vals >= _threshold(vals, method)
    return adata


def _threshold(vals: np.ndarray, method: str) -> float:
    finite = np.asarray(vals, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or finite.min() == finite.max():
        return np.inf  # nothing is positive
    if method == "otsu":
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(finite))
    # default: 2-component GMM, decision boundary between the component means
    try:
        from sklearn.mixture import GaussianMixture

        g = GaussianMixture(n_components=2, random_state=0).fit(finite.reshape(-1, 1))
        lo, hi = sorted(g.means_.ravel())
        if hi <= lo:
            raise ValueError("degenerate GMM")
        hi_comp = int(np.argmax(g.means_.ravel()))
        xs = np.linspace(lo, hi, 256)
        post = g.predict_proba(xs.reshape(-1, 1))[:, hi_comp]
        return float(xs[min(int(np.searchsorted(post, 0.5)), xs.size - 1)])
    except Exception:
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(finite))


def gate(adata: ad.AnnData, gates: list[dict[str, Any]]) -> ad.AnnData:
    """Assign ``obs['phenotype']`` by hierarchical gates (first match wins).

    Each gate is ``{label, all?: [...], any?: [...], not?: [...]}`` over marker
    positivity. A gate with no condition is the default (applies to whatever is
    still unassigned)."""
    pheno = np.full(adata.n_obs, "Other", dtype=object)
    assigned = np.zeros(adata.n_obs, dtype=bool)

    def pos(m: str) -> np.ndarray:
        col = f"{m}_positive"
        if col not in adata.obs:
            raise KeyError(f"gate references {m!r} but {col} is missing; run mark_positive first")
        return adata.obs[col].to_numpy()

    for g in gates:
        if not any(k in g for k in ("all", "any", "not")):
            cond = ~assigned
        else:
            cond = np.ones(adata.n_obs, dtype=bool)
            for m in g.get("all", []):
                cond &= pos(m)
            if "any" in g:
                anyc = np.zeros(adata.n_obs, dtype=bool)
                for m in g["any"]:
                    anyc |= pos(m)
                cond &= anyc
            for m in g.get("not", []):
                cond &= ~pos(m)
            cond &= ~assigned
        pheno[cond] = g["label"]
        assigned |= cond

    adata.obs["phenotype"] = pd.Categorical(pheno)
    return adata


def phenotype(adata: ad.AnnData, config: PanelConfig) -> ad.AnnData:
    """Run positivity + gating from a ``config.phenotype`` block."""
    p = config.phenotype
    mark_positive(adata, p.get("markers"), method=p.get("positivity", {}).get("method", "gmm"))
    if p.get("gates"):
        gate(adata, p["gates"])
    return adata
