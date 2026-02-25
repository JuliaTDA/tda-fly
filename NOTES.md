# Notes: Methods that did NOT work

## Filtrations removed

### Directional (height) filtrations — REMOVED
- Swept a hyperplane across the wing along 8 angular directions (every 22.5°)
- **Problem**: The wing images are too noisy (isolated pixels, incomplete segmentation). Each sweep direction creates spurious topological features that do not correspond to actual vein geometry. Classifiers (especially LDA) exploit these noise patterns to separate classes, leading to inflated accuracy that does not generalize.
- The 8×(H0+H1) = 16 persistence diagrams added ~176 summary statistics, dramatically increasing the feature-to-sample ratio without genuine signal.

### EDT (Euclidean Distance Transform) filtration — REMOVED
- Assigned each foreground pixel its distance to the nearest background pixel; thick veins get high values.
- **Problem**: Since the images are binarized (only black and white pixels), the EDT produces nearly trivial persistence diagrams. Most veins are 1 pixel wide, so the EDT values are almost uniform. The resulting topological features carry no meaningful information about vein structure.

### Cubical (grayscale sublevel-set) filtration — REMOVED
- Computed sublevel-set persistence on the raw grayscale image.
- **Problem**: The wing images are already binarized (black veins on white background). With only two pixel values (0 and 1), the cubical filtration produces a single trivial step — all background appears at one threshold and all foreground at another. No meaningful topological information can be extracted from a binary image via grayscale sublevel sets.

## High-dimensional feature representations — NOT USED

### Persistence images + Betti curves (as classifier features)
- These produce high-dimensional vectors (100–225 features per diagram).
- With multiple filtrations × H0/H1, the combined feature matrix easily reaches ~1000 features for only 72 samples.
- **Problem**: Overfitting. LDA achieved 86% on 991 features / 72 samples, but this is likely noise exploitation. Nested LOOCV showed lower honest accuracy. Summary statistics (11–19 features per diagram) are sufficient and far more interpretable.

## Classifiers that were not effective

### SVM on distance matrices
- Converted persistence diagram distances to an RBF-like kernel and trained linear SVM.
- **Problem**: Performance was generally worse than direct k-NN on the same distances, with more hyperparameters to tune.

### Ensemble methods (majority/weighted voting)
- Combined k-NN, SVM, RF, LDA predictions.
- **Problem**: Adds complexity without clear improvement over the best single classifier when features are well-chosen. Not explainable.

## Key lesson

**Simple features + simple classifiers** (Rips + Radial summary statistics with LDA or Decision Tree) are more trustworthy and interpretable than complex pipelines with hundreds of features on a 72-sample dataset.
