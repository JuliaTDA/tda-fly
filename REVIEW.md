# Referee Report: *Diptera wing classification using Topological Data Analysis*

**Overall verdict**: Major revisions required. The methodology is generally sound but has several statistical weaknesses, the TDA content is too elementary for a specialized venue, and several claims are unsupported or technically incorrect.

---

## 1. Critical errors and incorrect statements

**H0 from Rips is not "a single infinite bar"** (Methods, Vietoris-Rips section)

The paper states: *"H0 is uninformative here because the connected point cloud yields a single infinite bar."* This is wrong. A Rips filtration on 750 farthest-point samples from a wing will produce 749 *finite* H0 bars (components merging) plus one infinite bar. The finite H0 bars encode connectivity at small scales and are not uninformative. The real reason to skip Rips H0 is likely that it is dominated by the sampling geometry rather than the vein topology, or that pilots showed it to be redundant with H1 — but that reason is not given.

**Pooled confusion matrix overstates precision**

The confusion matrix and `family_metrics` table are computed over `pooled_true` / `pooled_pred`, which contain 30 copies of each specimen (70 × 30 = 2100 "predictions"). Treating this as 2100 independent observations inflates precision and makes the heatmap visually crisper than the evidence warrants. The actual effective sample size per family is 3–12 specimens, not 90–360. This must be prominently flagged.

**Feature reduction threshold is circulary noisy**

The screen runs with `FEATURE_SCREEN_REPEATS = 10` while the full CV uses 30. The threshold for feature acceptance (`accuracy ≥ full_accuracy − 0.01`) is compared against estimates of different variance. Stochastic noise alone can cause a small feature set to appear "within tolerance" just because the 10-repeat estimate is volatile. This means the essential feature count is not reliably determined.

---

## 2. Missing non-TDA baseline

This is the most serious scientific gap. The paper claims TDA features are useful for classifying fly wings, but never compares against standard shape descriptors (Hu moments, Zernike moments, wing-cell area ratios, simple aspect ratios). Without a non-TDA baseline, the paper cannot support the implicit claim that TDA adds value over conventional morphometrics. A TDA-specialized journal will notice this immediately.

---

## 3. Unjustified hyperparameter choices

The preprocessing parameters are presented as facts without justification:

- **Blur = 1.8**: Why? What happens at 1.0 or 2.5?
- **Threshold = 0.08**: Same issue.
- **750 farthest-point samples**: No sensitivity analysis. Rips H1 can change qualitatively with sample size on vein networks with thin structures.
- **`cutoff = 5, threshold = 200`** in `rips_pd_1d`: Unexplained. "Cutoff" and "threshold" in the Rips context need mathematical definitions.

These choices are consequential — they determine which topological features exist — but they are treated as self-evident.

---

## 4. Wasserstein comparison is cherry-picked

The paper evaluates 4 Wasserstein variants × 6 classifiers = 24 combinations, then selects the best for comparison against the Random Forest. The "best result is close to the feature-based Random Forest" is therefore not surprising; it is guaranteed to be optimistic by the selection process. The comparison should either use a single pre-specified Wasserstein setup or apply a multiple-comparison correction.

---

## 5. Feature importance is not cross-validated

The "Feature importance" section computes importance from trees fit on the **full dataset** — not from the cross-validated trees. The paper acknowledges this but calls it "descriptive," then uses it as the basis for the feature reduction screen. If the importance ranking is biased, the essential feature set is biased. The reader deserves a stronger caveat and ideally a permutation-based or out-of-fold importance estimate.

---

## 6. Statistics presented without formulas

For a TDA journal, summary statistics need mathematical definitions:

- **Persistence entropy**: Which definition? Atienza et al. (2020)? Shannon entropy of normalized persistence values? The formula changes the value substantially.
- **Mean midlife**: Is this (birth + death)/2 per interval, then averaged? Or the midlife of the mean birth/death? The code says `(birth + death) / 2` per interval, but the text says "mean midlife" without clarification.
- **`pers_range`**: Defined as max − min persistence. But `max_pers` is already included, so this adds information only if min persistence is meaningful — which for Rips H1 it typically is not (the smallest bar is near zero).

---

## 7. No stability analysis

TDA persistence diagrams are theoretically stable under perturbations, but this is a theoretical guarantee under specific conditions (Lipschitz filtration functions, bounded diagrams). The specific pipeline — FPS subsampling, blur, threshold, connectivity correction — introduces discretization errors that break the stability guarantee. An empirical stability check (e.g., bootstrap re-samples of the 750-point FPS, or repeated runs with different random seeds for the sampling) is expected for a TDA paper.

---

## 8. The radial filtration lacks a citation

The radial filtration as defined here (distance-from-centroid sublevel sets on a binary image) is a specific design choice. Is this novel? If so, claim it. If not, cite it. As written, the reader cannot tell.

---

## 9. Callout box is too elementary

The persistent homology overview callout ("H0 records connected components; H1 records loops") is appropriate for a general biology journal, not a TDA venue. Either remove it or replace it with something substantive — e.g., a discussion of the stability theorem and what it implies for classification robustness.

---

## 10. Structural and writing issues

- **"Earlier versions of this analysis used..."** (Wasserstein section): A paper should not document its own revision history. Rewrite as a deliberate experimental design: "We include a pure metric-space baseline to assess how much signal is contained in the raw diagrams before feature extraction."
- **"Why some methods were not used"** (Conclusions): This is unusual and defensive. The information is useful but should be integrated into the Methods as a brief design rationale, not placed as a separate conclusion subsection.
- No actual numeric results appear in the Discussion or Conclusion sections — all sentences are qualitative. The reader should not have to hunt through rendered table outputs to find the key numbers.
- The Introduction has no literature review of TDA applied to morphology or wing shape analysis. There is prior work (e.g., persistent homology on leaf venation, insect wing geometry) that should be cited to position the contribution.

---

## Minor issues

| Location | Issue |
|---|---|
| Abstract | "macro-recall" is reported but the code variable is `balanced_accuracy` — clarify these are identical |
| Data section | No source or provenance for the 70 images — who collected them, under what conditions? |
| Code: `canonical_id` | Deduplication logic is silently embedded; 2 specimens with the same family+number but different images would be incorrectly merged |
| Code: `per_class_metrics` | Called on pooled data — the `support` column is 30× inflated, misleading readers who inspect the table |
| Bibliography | No citations visible in the rendered text anywhere in the Methods — all claims are unsupported |

---

## Summary of required revisions

1. Add a non-TDA morphometric baseline
2. Correct the H0 / "single infinite bar" claim
3. Flag explicitly that pooled metrics are over repeated folds, not independent specimens
4. Pre-specify a single Wasserstein variant or correct for multiple comparisons
5. Provide mathematical definitions for all summary statistics
6. Add sensitivity analyses for blur, FPS sample size, and threshold parameters
7. Cite prior work on radial filtrations or claim novelty
8. Add a proper literature review to the Introduction
