OTHERS_LABELS = [
    "Leiden+Wilcoxon",
    "scVI+Wilcoxon",
    "Harmony+Leiden+Wilcoxon",
    "Scanorama+Leiden+Wilcoxon",
    "LDVAE",
    "NMF",
    "scHPF",
]

OTHERS_RES_SWEEPS = dict(
    zip(
        OTHERS_LABELS,
        [
            [1.0, 0.3, 0.1, 0.03],
            [1.0, 0.3, 0.1, 0.03],
            [1.0, 0.3, 0.1, 0.03],
            [1.0, 0.3, 0.1, 0.03],
            [10, 3, 1, 0.3],
            [10, 3, 1, 0.3],
            [10, 3, 1, 0.3],
        ],
    )
)
