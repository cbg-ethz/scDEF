"""Save the HGSOC fits of the seeds that best-of-five did not keep, to check that a result holds across them.

``refit_best_of_seeds.py hgsoc`` keeps only the lowest-loss fit; this refits the other seeds with the same
workflow (``fit_fixed``, decision-fixed-workflow) and saves each as ``scdef-hgsoc-ref-seed{seed}``;
the fits of the earlier per-tutorial workflow are kept with the suffix ``-tutorialwf``.

    python reproducibility/make_figures/refit_hgsoc_other_seeds.py 42 1 2 3
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refit_best_of_seeds as rb  # noqa: E402

if __name__ == "__main__":
    a = rb.tutorial_adata("hgsoc")
    for seed in map(int, sys.argv[1:]):
        t0 = time.time()
        m = rb.fit_fixed(a.copy(), seed, batch_key=rb.BATCH_KEY["hgsoc"])
        print(rb.describe(m, seed, (time.time() - t0) / 60), flush=True)
        rb.save(m, f"scdef-hgsoc-ref-seed{seed}")
