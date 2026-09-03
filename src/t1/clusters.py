"""Working-cluster map from released cell-type labels (T1 tech plan §6.1)."""

from __future__ import annotations

# Official labels -> working cluster.
LABEL_TO_CLUSTER: dict[str, str] = {
    "aSHF": "SHF",
    "pSHF": "SHF",
    "OFT/RV-CM": "CM_OFT",
    "IFT-CM": "CM_IFT",
    "AVC-CM": "CM_AVC",
    "SV-CM": "CM_SV",
    "LV-CM": "CM_V",
    "RV-CM": "CM_V",
    "V-CM": "CM_V",
    "Endothelium": "endo",
    "Endocardium": "endo",
    "BEC": "endo",
    "Pericardium": "peri",
    "Proepicardium": "peri",
    "NCC": "ncc",
    "NCC-derived": "ncc",
    "Foregut": "gut",
    "Hepatocyte": "gut",
    "Surface Ectoderm": "ect",
    "Blood": "blood",
    "aPHM": "phm",
    "pPHM": "phm",
    "ST": "st_epi",
    "Epithelium": "st_epi",
    "Neural Tube": "dissect_out",
    "Paraxial Mesoderm": "dissect_out",
    "EXEM": "dissect_out",
    "JCF": "dissect_out",
}

CLUSTER_ORDER: tuple[str, ...] = (
    "SHF",
    "CM_OFT",
    "CM_IFT",
    "CM_AVC",
    "CM_SV",
    "CM_V",
    "endo",
    "peri",
    "ncc",
    "gut",
    "ect",
    "blood",
    "phm",
    "st_epi",
    "dissect_out",
)

BIRTH_CLUSTERS: frozenset[str] = frozenset({"phm", "st_epi"})
DISSECT_CLUSTER = "dissect_out"

# Birth clusters have no E8.5 source; one-step proxy samples from these progenitors.
BIRTH_PROGENITOR: dict[str, str] = {
    "phm": "SHF",
    "st_epi": "gut",
}

FLOW_CLUSTERS: tuple[str, ...] = tuple(
    k for k in CLUSTER_ORDER if k not in BIRTH_CLUSTERS and k != DISSECT_CLUSTER
)


def cluster_id_map() -> dict[str, int]:
    return {name: i for i, name in enumerate(CLUSTER_ORDER)}


def labels_to_clusters(labels) -> list[str]:
    out = []
    missing = set()
    for lab in labels:
        key = str(lab)
        if key not in LABEL_TO_CLUSTER:
            missing.add(key)
            out.append("dissect_out")
        else:
            out.append(LABEL_TO_CLUSTER[key])
    if missing:
        raise KeyError(f"Unmapped celltype labels: {sorted(missing)}")
    return out
