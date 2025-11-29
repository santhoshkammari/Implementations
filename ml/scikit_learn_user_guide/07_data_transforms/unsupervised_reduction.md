# 7.5. Unsupervised dimensionality reduction

If your number of features is high, it may be useful to reduce it with an unsupervised step prior to supervised steps. Many of the [Unsupervised learning](<../unsupervised_learning.html#unsupervised-learning>) methods implement a `transform` method that can be used to reduce the dimensionality. Below we discuss two specific examples of this pattern that are heavily used.

**Pipelining**

The unsupervised data reduction and the supervised estimator can be chained in one step. See [Pipeline: chaining estimators](<compose.html#pipeline>).

## 7.5.1. PCA: principal component analysis

[`decomposition.PCA`](<generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA> "sklearn.decomposition.PCA") looks for a combination of features that capture well the variance of the original features. See [Decomposing signals in components (matrix factorization problems)](<decomposition.html#decompositions>).

Examples

  * [Faces recognition example using eigenfaces and SVMs](<../auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py>)




## 7.5.2. Random projections

The module: [`random_projection`](<../api/sklearn.random_projection.html#module-sklearn.random_projection> "sklearn.random_projection") provides several tools for data reduction by random projections. See the relevant section of the documentation: [Random Projection](<random_projection.html#random-projection>).

Examples

  * [The Johnson-Lindenstrauss bound for embedding with random projections](<../auto_examples/miscellaneous/plot_johnson_lindenstrauss_bound.html#sphx-glr-auto-examples-miscellaneous-plot-johnson-lindenstrauss-bound-py>)




## 7.5.3. Feature agglomeration

[`cluster.FeatureAgglomeration`](<generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration> "sklearn.cluster.FeatureAgglomeration") applies [Hierarchical clustering](<clustering.html#hierarchical-clustering>) to group together features that behave similarly.

Examples

  * [Feature agglomeration vs. univariate selection](<../auto_examples/cluster/plot_feature_agglomeration_vs_univariate_selection.html#sphx-glr-auto-examples-cluster-plot-feature-agglomeration-vs-univariate-selection-py>)

  * [Feature agglomeration](<../auto_examples/cluster/plot_digits_agglomeration.html#sphx-glr-auto-examples-cluster-plot-digits-agglomeration-py>)




**Feature scaling**

Note that if features have very different scaling or statistical properties, [`cluster.FeatureAgglomeration`](<generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration> "sklearn.cluster.FeatureAgglomeration") may not be able to capture the links between related features. Using a [`preprocessing.StandardScaler`](<generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler> "sklearn.preprocessing.StandardScaler") can be useful in these settings.
