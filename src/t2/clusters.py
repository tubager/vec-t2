"""Working-cluster maps for T2 embryo and heart settings (tech plan §6.2–6.3)."""

from __future__ import annotations

from collections import Counter

# --- embryo -----------------------------------------------------------------

EMBRYO_LABEL_TO_CLUSTER: dict[str, str] = {
    "EXE-Endoderm": "exe_endo",
    "ExEM-1": "exem",
    "ExEM-2": "exem",
    "p-EXEM": "exem",
    "EXE-Ectoderm": "exe_ect",
    "Anterior Epiblast": "epi",
    "Caudal Epiblast": "epi",
    "Primitive Streak": "streak",
    "Gut Endoderm": "gut",
    "V-FG": "gut",
    "D-FG": "gut",
    "aSE": "se",
    "pSE": "se",
    "V-SE": "se",
    "PHM/PAM": "phm",
    "aPHM": "phm",
    "pPHM": "phm",
    "SOM": "som",
    "LPM": "lpm",
    "Allantois": "allantois",
    "Allantois Endothelium": "allantois",
    "HEM-Endoth": "hem",
    "Blood Progenitor": "hem",
    "Intra-Endoth": "hem",
    "EXE-Endothlium": "hem",  # spelling in E8.0
    "EXE-Endothelium": "hem",
    "CP": "heart",
    "FHF": "heart",
    "SHF": "heart",
    "JCF": "heart",
    "Forebrain": "neural",
    "Hindbrain": "neural",
    "d-CSE": "neural",
    "Unknown": "unknown",
}

EMBRYO_CLUSTER_ORDER: tuple[str, ...] = (
    "exe_endo",
    "exem",
    "exe_ect",
    "epi",
    "streak",
    "gut",
    "se",
    "phm",
    "som",
    "lpm",
    "allantois",
    "hem",
    "heart",
    "neural",
    "unknown",
)

# Born at E8.0; interpolation should draw from the right anchor, not amplify Unknown.
EMBRYO_BIRTH_CLUSTERS: frozenset[str] = frozenset({"neural"})
EMBRYO_DEATH_CLUSTERS: frozenset[str] = frozenset({"exe_ect"})
EMBRYO_BIRTH_PROGENITOR: dict[str, str] = {"neural": "epi"}

# --- heart ------------------------------------------------------------------

HEART_LABEL_TO_CLUSTER: dict[str, str] = {
    "V-CM": "CM_V",
    "IFT-CM": "CM_IFT",
    "A-CM": "CM_IFT",
    "SV-CM": "CM_IFT",
    "OFT-CM": "CM_OFT",
    "Endo": "endo",
    "Intra-Endoth-1": "endo",
    "Intra-Endoth-2": "endo",
    "Chamber-Endo": "endo",
    "Cush-EndoMT-Endo": "endo",
    "BEC": "endo",
    "early-VEC": "endo",
    "Great Artery Endoth": "endo",
    "Peri": "peri",
    "Proepi": "peri",
    "NCC": "ncc",
    "aPHM": "phm",
    "pPHM": "phm",
    "PAM-1": "pam",
    "PAM-2": "pam",
    "PAM-3": "pam",
    "PAM-4": "pam",
    "Dorsal-PAM": "pam",
    "Lateral-PAM": "pam",
    "pSE": "se",
    "V-CSE": "se",
    "d-CSE": "se",
    "SE": "se",
    "D-FG": "gut",
    "a-FG": "gut",
    "Lateral FG": "gut",
    "Gut Endoderm": "gut",
    "Hindgut": "gut",
    "Foregut": "gut",
    "Hepatocytes": "gut",
    "Neural Tube": "nt",
    "Forebrain": "nt",
    "EXE-Endoderm": "extra",
    "ExEM-1": "extra",
    "ExEM-2": "extra",
    "Allantois": "extra",
    "HEM-Endoth": "extra",
    "SOM": "extra",
    "LPM": "extra",
    "Caudal Epiblast": "extra",
    "JCF": "extra",
    "Unknown": "extra",
    "Branch Arch": "branch",
    "ST": "st",
    "DMP": "st",
}

HEART_CLUSTER_ORDER: tuple[str, ...] = (
    "CM_V",
    "CM_IFT",
    "CM_OFT",
    "endo",
    "peri",
    "ncc",
    "phm",
    "pam",
    "se",
    "gut",
    "nt",
    "extra",
    "branch",
    "st",
)

# FOV-lost at t>=9.5: keep at 0, do not grow back from E8.25.
HEART_FOV_LOST: frozenset[str] = frozenset({"extra", "nt"})
HEART_BIRTH_CLUSTERS: frozenset[str] = frozenset({"CM_OFT", "branch", "st"})
HEART_BIRTH_PROGENITOR: dict[str, str] = {
    "CM_OFT": "CM_IFT",
    "branch": "ncc",
    "st": "peri",
}


def cluster_order(setting: str) -> tuple[str, ...]:
    if setting == "embryo":
        return EMBRYO_CLUSTER_ORDER
    if setting == "heart":
        return HEART_CLUSTER_ORDER
    raise ValueError(f"Unknown setting {setting}")


def label_map(setting: str) -> dict[str, str]:
    if setting == "embryo":
        return EMBRYO_LABEL_TO_CLUSTER
    if setting == "heart":
        return HEART_LABEL_TO_CLUSTER
    raise ValueError(f"Unknown setting {setting}")


def birth_clusters(setting: str) -> frozenset[str]:
    if setting == "embryo":
        return EMBRYO_BIRTH_CLUSTERS
    if setting == "heart":
        return HEART_BIRTH_CLUSTERS
    raise ValueError(f"Unknown setting {setting}")


def fov_lost_clusters(setting: str) -> frozenset[str]:
    if setting == "heart":
        return HEART_FOV_LOST
    return frozenset()


def birth_progenitor(setting: str) -> dict[str, str]:
    if setting == "embryo":
        return EMBRYO_BIRTH_PROGENITOR
    if setting == "heart":
        return HEART_BIRTH_PROGENITOR
    return {}


def labels_to_clusters(labels, setting: str) -> list[str]:
    mapping = label_map(setting)
    out = []
    missing: set[str] = set()
    for lab in labels:
        key = str(lab)
        if key not in mapping:
            missing.add(key)
            out.append("unknown" if setting == "embryo" else "extra")
        else:
            out.append(mapping[key])
    if missing:
        raise KeyError(f"Unmapped {setting} celltype labels: {sorted(missing)}")
    return out


def cluster_id_map(setting: str) -> dict[str, int]:
    return {name: i for i, name in enumerate(cluster_order(setting))}


def shared_flow_clusters(
    labels_left,
    labels_right,
    setting: str,
    min_cells: int = 16,
) -> list[str]:
    """Working clusters present on both sides of a hop. Birth clusters are excluded."""
    left = labels_to_clusters(labels_left, setting)
    right = labels_to_clusters(labels_right, setting)
    n0, n1 = Counter(left), Counter(right)
    birth = birth_clusters(setting)
    out = []
    for k in cluster_order(setting):
        if k in birth:
            continue
        if n0.get(k, 0) >= min_cells and n1.get(k, 0) >= min_cells:
            out.append(k)
    return out
