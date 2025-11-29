# machine learning mastery through sklearn
## efficient learning path for production-ready ML

---

## philosophy

you already know ML concepts. goal = clean understanding + deep math + production capability.

**100 focused hours > 10000 wandering hours**

this roadmap is designed for:
- applied AI research engineer mindset
- concepts that stick
- math that makes sense
- code that ships

---

## PART 1: CLASSIFICATION & PRIORITY

### TIER 0: ABSOLUTE FOUNDATIONS (must master - no shortcuts)
**why:** everything else builds on this. weak foundation = permanent confusion.

**from supervised learning:**
- 1.1.1 - Ordinary Least Squares (OLS)
- 1.1.2 - Ridge regression
- 1.1.3 - Lasso
- 1.1.11 - Logistic regression
- 1.4 - Support Vector Machines (full section - all subsections)
  - classification, regression, kernel functions, mathematical formulation

**from unsupervised learning:**
- 2.5.1 - Principal Component Analysis (PCA)

**from data transforms:**
- 7.3 - Preprocessing data (standardization, normalization, encoding)
- 7.1.1 - Pipeline: chaining estimators

**from model selection:**
- 3.1 - Cross-validation (full section)
- 3.4.4 - Classification metrics
- 3.4.6 - Regression metrics

**from common pitfalls:**
- 11.2 - Data leakage (CRITICAL - read this early)
- 11.1 - Inconsistent preprocessing

**time estimate:** 20-25 hours
**mastery check:** can you derive the closed-form solution for Ridge? explain kernel trick without looking? implement cross-validation from scratch?

---

### TIER 1: PRODUCTION WORKHORSES (where real ML happens)
**why:** these are the algorithms you'll use 80% of the time in production.

**from supervised learning:**
- 1.10 - Decision Trees (full section)
  - classification, regression, complexity, CART algorithm, pruning
- 1.11 - Ensembles (full section - this is crucial)
  - gradient boosting, random forests, bagging, stacking
  - understand bias-variance tradeoff here
- 1.13 - Feature selection (full section)
- 1.1.13 - Stochastic Gradient Descent (SGD)

**from unsupervised learning:**
- 2.3 - Clustering (focus on K-means, DBSCAN, HDBSCAN)

**from model selection:**
- 3.2 - Hyperparameter tuning
  - grid search, random search, successive halving
- 3.5 - Validation curves

**from data transforms:**
- 7.4 - Imputation of missing values
- 7.1.4 - ColumnTransformer for heterogeneous data

**time estimate:** 25-30 hours
**mastery check:** can you explain why random forests reduce variance? when to use gradient boosting vs random forest? implement a simple decision tree from scratch?

---

### TIER 2: SPECIALIZED TECHNIQUES (expand your toolkit)
**why:** these handle specific scenarios you'll encounter in real projects.

**from supervised learning:**
- 1.1.5 - Elastic-Net (combines L1 + L2)
- 1.1.16 - Robust regression (outliers and modeling errors)
- 1.1.17 - Quantile regression
- 1.6 - Nearest Neighbors (KNN, algorithms, complexity)
- 1.9 - Naive Bayes (fast, works well for text)
- 1.16 - Probability calibration
- 1.12 - Multiclass and multioutput algorithms

**from unsupervised learning:**
- 2.2.9 - t-SNE (visualization)
- 2.5 - Matrix factorization (SVD, NMF, ICA, LDA)
- 2.7 - Outlier detection

**from data transforms:**
- 7.5 - Dimensionality reduction (PCA, random projections)
- 7.6 - Random Projection
- 7.7 - Kernel Approximation

**time estimate:** 20-25 hours
**mastery check:** know when to use each technique. understand tradeoffs.

---

### TIER 3: ADVANCED & NICHE (learn when needed)
**why:** valuable for specific use cases, but not daily drivers.

**from supervised learning:**
- 1.1.10 - Bayesian Regression
- 1.1.7 - Least Angle Regression (LARS)
- 1.2 - LDA/QDA (Linear/Quadratic Discriminant Analysis)
- 1.3 - Kernel ridge regression
- 1.7 - Gaussian Processes
- 1.8 - Cross decomposition (PLS)
- 1.14 - Semi-supervised learning
- 1.15 - Isotonic regression
- 1.17 - Neural networks (sklearn's MLP - use pytorch/tensorflow instead)

**from unsupervised learning:**
- 2.1 - Gaussian mixture models
- 2.2 - Manifold learning (Isomap, LLE, etc - mostly research)
- 2.4 - Biclustering
- 2.6 - Covariance estimation
- 2.8 - Density estimation
- 2.9 - Restricted Boltzmann machines (outdated)

**from model selection:**
- 3.3 - Decision threshold tuning

**from data transforms:**
- 7.2 - Feature extraction (text, images)

**time estimate:** 15-20 hours (learn on-demand)
**approach:** reference material. read when you encounter the specific problem.

---

### TIER 4: INFRASTRUCTURE & UTILITIES (skim once, reference later)
**why:** important to know they exist, but you'll look them up when needed.

- 4 - Metadata routing (boring, check when needed)
- 5 - Inspection (partial dependence, permutation importance)
- 6 - Visualizations (check examples when needed)
- 8 - Datasets (toy datasets, loaders - trivial)
- 9 - Computing (parallelism, performance tips)
- 10 - Model persistence (pickling, ONNX, joblib)
- 12 - Dispatching (array API - niche)
- 13 - Estimator selection (flowchart)
- 14 - External resources (videos, tutorials)

**time estimate:** 5-10 hours (skim)
**approach:** awareness only. deep dive when you need it.

---

### TIER 5: SKIP ENTIRELY (not worth your time)
**why:** outdated, replaced by better tools, or too niche.

**skip these:**
- 1.1.4 - Multi-task Lasso (rarely used)
- 1.1.6 - Multi-task Elastic-Net (rarely used)
- 1.1.8 - LARS Lasso (academic curiosity)
- 1.1.9 - Orthogonal Matching Pursuit (signal processing niche)
- 1.1.14 - Perceptron (use logistic regression)
- 1.1.15 - Passive Aggressive Algorithms (online learning - use SGD)
- 1.17 - sklearn neural networks (use pytorch/jax/tensorflow)
- 2.2.3-2.2.7 - exotic manifold methods (use t-SNE or UMAP)
- 2.9 - RBM (dead approach, use modern deep learning)

**rationale:** these either have better alternatives or solve problems you won't encounter in production work.

---

## PART 2: LEARNING SEQUENCE

### PHASE 1: FOUNDATIONS (weeks 1-3)
**goal:** rock-solid understanding of linear methods, kernels, evaluation

**week 1: linear regression & regularization**
- OLS → Ridge → Lasso
- understand L1 vs L2 penalties
- derive solutions mathematically
- implement from scratch
- common pitfalls: data leakage, inconsistent preprocessing

**week 2: classification fundamentals**
- logistic regression (derive gradient descent)
- SVM (linear → kernel trick → dual formulation)
- understand decision boundaries
- metrics: precision, recall, F1, ROC-AUC

**week 3: dimensionality reduction & validation**
- PCA (eigendecomposition, variance explained)
- cross-validation (k-fold, stratified, time series)
- pipelines (fit_transform vs transform)
- practical preprocessing

**deliverable:** implement Ridge, Logistic, PCA from scratch. run experiments on real dataset.

---

### PHASE 2: PRODUCTION ALGORITHMS (weeks 4-7)
**goal:** master the tools you'll actually use in production

**week 4: decision trees**
- CART algorithm (Gini vs entropy)
- overfitting & pruning
- feature importance
- handling missing values

**week 5: random forests**
- bagging concept
- out-of-bag error
- feature randomness
- when to use vs gradient boosting

**week 6-7: gradient boosting**
- boosting vs bagging
- loss functions
- learning rate & trees
- XGBoost/LightGBM/CatBoost (beyond sklearn)
- **this is where you'll spend most time in production**

**deliverable:** kaggle competition or real project using ensembles. compare RF vs GBM.

---

### PHASE 3: COMPLETE THE TOOLKIT (weeks 8-10)
**goal:** expand techniques for specific scenarios

**week 8: clustering & unsupervised**
- k-means (lloyd's algorithm, k-means++, elbow method)
- DBSCAN (density-based, no need to specify k)
- evaluation metrics (silhouette, calinski-harabasz)

**week 9: feature engineering**
- feature selection (RFE, SelectFromModel)
- missing value imputation
- categorical encoding (one-hot, target, ordinal)
- polynomial features

**week 10: specialized regression**
- quantile regression (when you need prediction intervals)
- robust regression (outliers)
- elastic net (when you want both L1 + L2)

**deliverable:** end-to-end ML pipeline on messy real-world data

---

### PHASE 4: ADVANCED TOPICS (weeks 11-12)
**goal:** handle edge cases and advanced scenarios

- hyperparameter optimization (bayesian optimization with optuna)
- model calibration (when probabilities matter)
- imbalanced data strategies
- multioutput & multilabel problems
- outlier detection (isolation forest, LOF)

**deliverable:** production-ready model with proper validation, calibration, monitoring

---

## PART 3: STUDY METHOD

### for each topic:

**1. theory first (40% of time)**
- read sklearn docs section
- derive math on paper
- understand assumptions & limitations
- check wikipedia for deeper math
- find 1-2 good blog posts

**2. implement from scratch (30% of time)**
- numpy only
- don't look at sklearn source initially
- validates understanding
- reveals edge cases

**3. practical sklearn usage (20% of time)**
- learn the API properly
- understand parameters
- common gotchas
- best practices

**4. real application (10% of time)**
- apply to actual dataset
- compare with other methods
- understand when it fails
- document learnings

### resources to use alongside sklearn docs:

**math:**
- "Elements of Statistical Learning" (ESL) - the bible
- "Pattern Recognition and Machine Learning" (Bishop)
- 3Blue1Brown videos for linear algebra intuition

**practical:**
- kaggle competitions (learn from kernels)
- UCI ML repository for datasets
- your own projects

**implementation:**
- read sklearn source code (it's well-written)
- numpy documentation
- linear algebra review (Gilbert Strang lectures)

---

## PART 4: KEY CONCEPTS TO MASTER

### mathematical foundations:
- linear algebra: matrix operations, eigenvalues, SVD
- calculus: gradients, optimization, convexity
- probability: distributions, maximum likelihood, bayes theorem
- statistics: hypothesis testing, confidence intervals

### ML fundamentals:
- bias-variance tradeoff (THE central concept)
- overfitting vs underfitting
- regularization (L1, L2, elastic net)
- loss functions & optimization
- gradient descent variants

### practical skills:
- train/val/test splits (and why)
- cross-validation strategies
- hyperparameter tuning
- feature engineering
- handling imbalanced data
- model evaluation & selection
- pipeline construction
- debugging ML models

---

## PART 5: COMMON MISTAKES TO AVOID

### from common_pitfalls section:

**CRITICAL ERRORS:**
1. **data leakage** - fitting on test data
   - always split BEFORE preprocessing
   - use pipelines to prevent this

2. **inconsistent preprocessing** - different transforms for train/test
   - fit on train, transform on both
   - pipelines solve this

3. **wrong cross-validation** - using wrong splitter
   - time series: use TimeSeriesSplit
   - imbalanced: use StratifiedKFold

4. **random_state confusion** - not understanding randomness
   - use integers for reproducibility
   - use instances for robust CV

5. **test set reuse** - tuning on test set
   - use proper train/val/test split
   - or nested cross-validation

6. **ignoring data distribution** - train/test mismatch
   - check for distribution shift
   - validate assumptions

---

## PART 6: MASTERY CHECKPOINTS

### after tier 0 (foundations):
- [ ] can derive Ridge regression closed-form solution
- [ ] can explain kernel trick intuitively
- [ ] can implement k-fold CV from scratch
- [ ] understand when to use L1 vs L2 regularization
- [ ] can debug data leakage in pipelines

### after tier 1 (production):
- [ ] can explain bias-variance tradeoff with examples
- [ ] know when to use RF vs GBM vs linear models
- [ ] can tune hyperparameters efficiently
- [ ] understand tree-based feature importance
- [ ] can build end-to-end sklearn pipeline

### after tier 2 (specialized):
- [ ] know which algorithm for which problem type
- [ ] understand trade-offs between methods
- [ ] can handle missing data properly
- [ ] can detect and handle outliers
- [ ] understand probability calibration

### production-ready checklist:
- [ ] proper train/val/test splitting
- [ ] no data leakage (validated)
- [ ] reproducible results (random_state management)
- [ ] appropriate metrics for problem
- [ ] cross-validation for model selection
- [ ] hyperparameter tuning done properly
- [ ] model persistence strategy
- [ ] monitoring & debugging approach

---

## PART 7: WHY THIS ORDER?

### tier 0 first because:
- linear models → closed-form solutions → clean math
- understand regularization before trees
- SVMs teach kernel methods properly
- PCA is foundation for dimensionality reduction
- proper evaluation prevents bad conclusions
- data leakage will ruin everything if not learned early

### tier 1 after foundations:
- trees & ensembles are production workhorses
- need linear model intuition first
- gradient boosting requires understanding of loss functions
- feature selection needs understanding of models

### tier 2 when comfortable:
- these solve specific problems
- easier to learn with strong foundation
- can be learned on-demand

### tier 3 as reference:
- diminishing returns
- learn when problem requires it
- foundation makes these easy to pick up

---

## FINAL NOTES

this roadmap is **aggressive but achievable** if you:
- already know ML concepts (you do)
- focus on understanding, not memorization
- implement from scratch
- work through real problems
- skip the fluff

**total time estimate:** 85-110 hours of focused work
- tier 0: 20-25h
- tier 1: 25-30h
- tier 2: 20-25h
- tier 3: 15-20h (on-demand)
- tier 4: 5-10h (skim)

this gets you **production-ready ML mastery**, not academic knowledge.

you won't know every sklearn function, but you'll:
- understand the math deeply
- know which tool for which job
- avoid common pitfalls
- ship models that work
- debug when things break

**that's the goal.**

---

## NEXT STEPS

1. read this entire roadmap
2. agree/disagree/modify based on your needs
3. start with tier 0, week 1
4. i'll guide you through each topic with:
   - clean math explanations
   - practical examples
   - common mistakes
   - implementation tips
   - production insights

let's make you dangerous in ML.
