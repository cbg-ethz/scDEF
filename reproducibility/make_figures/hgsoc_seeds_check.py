"""Do the HGSOC treatment results hold in the fits that best-of-five did not keep?

Reads the outputs of ``hgsoc_treatment.py`` for the kept fit (``out/hgsoc_treatment_ref.json``) and for
the other seeds (``out/seeds_check/hgsoc_treatment_scdef-hgsoc-ref-seed*.json``, from the models saved by
``refit_hgsoc_other_seeds.py``). Factor numbers differ between fits, so a programme is matched by its
marker genes: a layer-0 factor carries it when at least two of the markers are in its top 10 genes.

    python reproducibility/make_figures/hgsoc_seeds_check.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMMES = {
    "immediate early (post)": ["JUN", "FOS", "GADD45B", "IER2", "JUNB", "EGR1"],
    "heat shock (post)": ["HSPA1A", "HSPA1B", "DNAJB1", "HSPA6"],
    "MHC class II (post)": ["CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1"],
    "CLU, IGFBP7, GPX3 (post)": ["CLU", "IGFBP7", "GPX3", "FN1", "CRYAB", "RARRES1"],
    "SAA1, SAA2 (post)": ["SAA1", "SAA2", "PSCA", "PERP"],
    "ER chaperones (post)": ["HSPA5", "HSP90B1", "HERPUD1", "PDIA4", "PDIA6", "CALR"],   # not SDF2L1: with HSP90B1 it also matches an FTL/APOE factor
    "metallothioneins (post)": ["MT2A", "MT1E", "MT1X", "MT1G"],
    "proliferation (naive)": ["HIST1H4C", "UBE2C", "STMN1", "TOP2A", "MKI67", "CENPF", "PTTG1"],
    "interferon (naive)": ["IFI27", "ISG15", "IFI6", "IFIT3", "CXCL10"],
    "epithelial (naive)": ["WFDC2", "MUC1", "SOX17", "MDK"],
}
# The layer-2 states of the text (kept fit after the initialisation fix, 2026-11-04), matched the same way among
# layer-2 factors. Technical factors are not excluded in the other fits (their names differ between fits).
STATES = {
    "stress-associated (up)": ["JUN", "FOS", "GADD45B", "IER2", "JUNB", "EGR1"],
    "IFN-gamma, MHC class II (up)": ["CD74", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "CLU", "RARRES1"],   # in other fits CD74 with CLU
    "SAA1, SAA2 (up)": ["SAA1", "SAA2", "PSCA", "PERP"],
    "S100A6, SCGB1D2 (up)": ["S100A6", "S100A4", "SCGB1D2", "SCGB2A1", "SRD5A3", "DEFB1"],
    "proliferative (down)": ["HIST1H4C", "UBE2C", "STMN1", "TOP2A", "MKI67", "TUBA1B"],
    "epithelial (down)": ["WFDC2", "MUC1", "SOX17", "MDK"],
    "interferon (no change)": ["IFI27", "ISG15", "IFI6", "IFIT3", "CXCL10", "LCN2", "S100A8", "S100A9"],
}


def states(r):
    """Each layer-2 state: the layer-2 factors carrying it, with the patients in which its weight rises / falls."""
    sh = r.get("L2_shift", {})
    out = {}
    for name, markers in STATES.items():
        out[name] = [dict(factor=f, up=int(sh["higher_after"][f]), down=int(sh["lower_after"][f]))
                     for f, genes in r["top_genes"].items()
                     if f.startswith("L2_") and len(set(genes) & set(markers)) >= 2 and f in sh.get("higher_after", {})]
    return out


def programmes(r):
    out = {}
    for name, markers in PROGRAMMES.items():
        hits = []
        for f, genes in r["top_genes"].items():
            if f.startswith("L0_") and len(set(genes) & set(markers)) >= 2:
                s = r["L0"]["summary"][f]
                call = "post" if f in r["L0"]["post"] else "naive" if f in r["L0"]["naive"] else "-"
                hits.append(dict(factor=f, call=call, up=int(s["up"]), down=int(s["down"]), tested=int(s["tested"])))
        out[name] = hits
    return out


def main():
    kept = json.load(open(os.path.join(HERE, "out", "refit_best_of_seeds_hgsoc.json")))["ref_chosen_seed"]
    files = {f"seed {kept} (kept)": os.path.join(HERE, "out", "hgsoc_treatment_ref.json")}
    for p in sorted(glob.glob(os.path.join(HERE, "out", "seeds_check", "hgsoc_treatment_scdef-hgsoc-ref-seed*.json"))):
        seed = int(p.rsplit("seed", 1)[1].split(".")[0])
        if seed != kept:                 # the kept seed is the best-of-five model itself
            files[f"seed {seed}"] = p
    summary = {}
    for label, p in files.items():
        r = json.load(open(p))
        rec = [t for t in r["transitions"] if t["patients_tested"] >= 4
               and t["patients_post_further"] == t["patients_tested"]]
        clu = {h["factor"] for h in programmes(r)["CLU, IGFBP7, GPX3 (post)"]}
        to_clu = [(t["nodes"], t["patients_post_further"], t["patients_tested"]) for t in r["transitions"] if t["nodes"][-1] in clu]
        summary[label] = dict(programmes=programmes(r), states_L2=states(r),
                              n_post=len(r["L0"]["post"]), n_naive=len(r["L0"]["naive"]),
                              paths_all_patients_min4=[(t["nodes"], t["patients_tested"]) for t in rec],
                              paths_to_clu_factor=to_clu)
    with open(os.path.join(HERE, "out", "hgsoc_seeds_check.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for label, s in summary.items():
        print(f"== {label}: {s['n_post']} post-enriched, {s['n_naive']} naive-enriched layer-0 factors")
        for name, hits in s["programmes"].items():
            print(f"  {name:26s}", "; ".join(f"{h['factor']} {h['call']} ({h['up']}/{h['down']}/{h['tested']})" for h in hits) or "absent")
        for name, hits in s["states_L2"].items():
            print(f"  L2 {name:26s}", "; ".join(f"{h['factor']} up {h['up']} down {h['down']}" for h in hits) or "absent")
        print("  paths with every tested patient (>= 4) shifted:", s["paths_all_patients_min4"] or "none")
        print("  paths to the CLU factor (further / tested):", [(" ".join(n), f"{a}/{b}") for n, a, b in s["paths_to_clu_factor"]] or "none")


if __name__ == "__main__":
    main()
