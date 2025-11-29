# 4\. Metadata Routing

Note

The Metadata Routing API is experimental, and is not yet implemented for all estimators. Please refer to the list of supported and unsupported models for more information. It may change without the usual deprecation cycle. By default this feature is not enabled. You can enable it by setting the `enable_metadata_routing` flag to `True`:
    
    
    >>> import sklearn
    >>> sklearn.set_config(enable_metadata_routing=True)
    

Note that the methods and requirements introduced in this document are only relevant if you want to pass [metadata](<glossary.html#term-metadata>) (e.g. `sample_weight`) to a method. If you’re only passing `X` and `y` and no other parameter / metadata to methods such as [fit](<glossary.html#term-fit>), [transform](<glossary.html#term-transform>), etc., then you don’t need to set anything.

This guide demonstrates how [metadata](<glossary.html#term-metadata>) can be routed and passed between objects in scikit-learn. If you are developing a scikit-learn compatible estimator or meta-estimator, you can check our related developer guide: [Metadata Routing](<auto_examples/miscellaneous/plot_metadata_routing.html#sphx-glr-auto-examples-miscellaneous-plot-metadata-routing-py>).

Metadata is data that an estimator, scorer, or CV splitter takes into account if the user explicitly passes it as a parameter. For instance, [`KMeans`](<modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans> "sklearn.cluster.KMeans") accepts `sample_weight` in its `fit()` method and considers it to calculate its centroids. `classes` are consumed by some classifiers and `groups` are used in some splitters, but any data that is passed into an object’s methods apart from X and y can be considered as metadata. Prior to scikit-learn version 1.3, there was no single API for passing metadata like that if these objects were used in conjunction with other objects, e.g. a scorer accepting `sample_weight` inside a [`GridSearchCV`](<modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV> "sklearn.model_selection.GridSearchCV").

With the Metadata Routing API, we can transfer metadata to estimators, scorers, and CV splitters using [meta-estimators](<glossary.html#term-meta-estimators>) (such as [`Pipeline`](<modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") or [`GridSearchCV`](<modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV> "sklearn.model_selection.GridSearchCV")) or functions such as [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate") which route data to other objects. In order to pass metadata to a method like `fit` or `score`, the object consuming the metadata, must _request_ it. This is done via `set_{method}_request()` methods, where `{method}` is substituted by the name of the method that requests the metadata. For instance, estimators that use the metadata in their `fit()` method would use `set_fit_request()`, and scorers would use `set_score_request()`. These methods allow us to specify which metadata to request, for instance `set_fit_request(sample_weight=True)`.

For grouped splitters such as [`GroupKFold`](<modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold> "sklearn.model_selection.GroupKFold"), a `groups` parameter is requested by default. This is best demonstrated by the following examples.

## 4.1. Usage Examples

Here we present a few examples to show some common use-cases. Our goal is to pass `sample_weight` and `groups` through [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate"), which routes the metadata to [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV") and to a custom scorer made with [`make_scorer`](<modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer> "sklearn.metrics.make_scorer"), both of which _can_ use the metadata in their methods. In these examples we want to individually set whether to use the metadata within the different [consumers](<glossary.html#term-consumer>).

The examples in this section require the following imports and data:
    
    
    >>> import numpy as np
    >>> from sklearn.metrics import make_scorer, accuracy_score
    >>> from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    >>> from sklearn.model_selection import cross_validate, GridSearchCV, GroupKFold
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.pipeline import make_pipeline
    >>> n_samples, n_features = 100, 4
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(n_samples, n_features)
    >>> y = rng.randint(0, 2, size=n_samples)
    >>> my_groups = rng.randint(0, 10, size=n_samples)
    >>> my_weights = rng.rand(n_samples)
    >>> my_other_weights = rng.rand(n_samples)
    

### 4.1.1. Weighted scoring and fitting

The splitter used internally in [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV"), [`GroupKFold`](<modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold> "sklearn.model_selection.GroupKFold"), requests `groups` by default. However, we need to explicitly request `sample_weight` for it and for our custom scorer by specifying `sample_weight=True` in [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV")’s `set_fit_request()` method and in [`make_scorer`](<modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer> "sklearn.metrics.make_scorer")’s `set_score_request()` method. Both [consumers](<glossary.html#term-consumer>) know how to use `sample_weight` in their `fit()` or `score()` methods. We can then pass the metadata in [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate") which will route it to any active consumers:
    
    
    >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
    >>> lr = LogisticRegressionCV(
    ...     cv=GroupKFold(),
    ...     scoring=weighted_acc
    ... ).set_fit_request(sample_weight=True)
    >>> cv_results = cross_validate(
    ...     lr,
    ...     X,
    ...     y,
    ...     params={"sample_weight": my_weights, "groups": my_groups},
    ...     cv=GroupKFold(),
    ...     scoring=weighted_acc,
    ... )
    

Note that in this example, [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate") routes `my_weights` to both the scorer and [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV").

If we would pass `sample_weight` in the params of [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate"), but not set any object to request it, `UnsetMetadataPassedError` would be raised, hinting to us that we need to explicitly set where to route it. The same applies if `params={"sample_weights": my_weights, ...}` were passed (note the typo, i.e. `weights` instead of `weight`), since `sample_weights` was not requested by any of its underlying objects.

### 4.1.2. Weighted scoring and unweighted fitting

When passing metadata such as `sample_weight` into a [router](<glossary.html#term-router>) ([meta-estimators](<glossary.html#term-meta-estimators>) or routing function), all `sample_weight` [consumers](<glossary.html#term-consumer>) require weights to be either explicitly requested or explicitly not requested (i.e. `True` or `False`). Thus, to perform an unweighted fit, we need to configure [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV") to not request sample weights, so that [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate") does not pass the weights along:
    
    
    >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
    >>> lr = LogisticRegressionCV(
    ...     cv=GroupKFold(), scoring=weighted_acc,
    ... ).set_fit_request(sample_weight=False)
    >>> cv_results = cross_validate(
    ...     lr,
    ...     X,
    ...     y,
    ...     cv=GroupKFold(),
    ...     params={"sample_weight": my_weights, "groups": my_groups},
    ...     scoring=weighted_acc,
    ... )
    

If [`linear_model.LogisticRegressionCV.set_fit_request`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV.set_fit_request> "sklearn.linear_model.LogisticRegressionCV.set_fit_request") had not been called, [`cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate") would raise an error because `sample_weight` is passed but [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV") would not be explicitly configured to recognize the weights.

### 4.1.3. Unweighted feature selection

Routing metadata is only possible if the object’s method knows how to use the metadata, which in most cases means they have it as an explicit parameter. Only then we can set request values for metadata using `set_fit_request(sample_weight=True)`, for instance. This makes the object a [consumer](<glossary.html#term-consumer>).

Unlike [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV"), [`SelectKBest`](<modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest> "sklearn.feature_selection.SelectKBest") can’t consume weights and therefore no request value for `sample_weight` on its instance is set and `sample_weight` is not routed to it:
    
    
    >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
    >>> lr = LogisticRegressionCV(
    ...     cv=GroupKFold(), scoring=weighted_acc,
    ... ).set_fit_request(sample_weight=True)
    >>> sel = SelectKBest(k=2)
    >>> pipe = make_pipeline(sel, lr)
    >>> cv_results = cross_validate(
    ...     pipe,
    ...     X,
    ...     y,
    ...     cv=GroupKFold(),
    ...     params={"sample_weight": my_weights, "groups": my_groups},
    ...     scoring=weighted_acc,
    ... )
    

### 4.1.4. Different scoring and fitting weights

Despite [`make_scorer`](<modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer> "sklearn.metrics.make_scorer") and [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV") both expecting the key `sample_weight`, we can use aliases to pass different weights to different consumers. In this example, we pass `scoring_weight` to the scorer, and `fitting_weight` to [`LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV"):
    
    
    >>> weighted_acc = make_scorer(accuracy_score).set_score_request(
    ...    sample_weight="scoring_weight"
    ... )
    >>> lr = LogisticRegressionCV(
    ...     cv=GroupKFold(), scoring=weighted_acc,
    ... ).set_fit_request(sample_weight="fitting_weight")
    >>> cv_results = cross_validate(
    ...     lr,
    ...     X,
    ...     y,
    ...     cv=GroupKFold(),
    ...     params={
    ...         "scoring_weight": my_weights,
    ...         "fitting_weight": my_other_weights,
    ...         "groups": my_groups,
    ...     },
    ...     scoring=weighted_acc,
    ... )
    

## 4.2. API Interface

A [consumer](<glossary.html#term-consumer>) is an object (estimator, meta-estimator, scorer, splitter) which accepts and uses some [metadata](<glossary.html#term-metadata>) in at least one of its methods (for instance `fit`, `predict`, `inverse_transform`, `transform`, `score`, `split`). Meta-estimators which only forward the metadata to other objects (child estimators, scorers, or splitters) and don’t use the metadata themselves are not consumers. (Meta-)Estimators which route metadata to other objects are [routers](<glossary.html#term-router>). A(n) (meta-)estimator can be a [consumer](<glossary.html#term-consumer>) and a [router](<glossary.html#term-router>) at the same time. (Meta-)Estimators and splitters expose a `set_{method}_request` method for each method which accepts at least one metadata. For instance, if an estimator supports `sample_weight` in `fit` and `score`, it exposes `estimator.set_fit_request(sample_weight=value)` and `estimator.set_score_request(sample_weight=value)`. Here `value` can be:

  * `True`: method requests a `sample_weight`. This means if the metadata is provided, it will be used, otherwise no error is raised.

  * `False`: method does not request a `sample_weight`.

  * `None`: router will raise an error if `sample_weight` is passed. This is in almost all cases the default value when an object is instantiated and ensures the user sets the metadata requests explicitly when a metadata is passed. The only exception are `Group*Fold` splitters.

  * `"param_name"`: alias for `sample_weight` if we want to pass different weights to different consumers. If aliasing is used the meta-estimator should not forward `"param_name"` to the consumer, but `sample_weight` instead, because the consumer will expect a param called `sample_weight`. This means the mapping between the metadata required by the object, e.g. `sample_weight` and the variable name provided by the user, e.g. `my_weights` is done at the router level, and not by the consuming object itself.




Metadata are requested in the same way for scorers using `set_score_request`.

If a metadata, e.g. `sample_weight`, is passed by the user, the metadata request for all objects which potentially can consume `sample_weight` should be set by the user, otherwise an error is raised by the router object. For example, the following code raises an error, since it hasn’t been explicitly specified whether `sample_weight` should be passed to the estimator’s scorer or not:
    
    
    >>> param_grid = {"C": [0.1, 1]}
    >>> lr = LogisticRegression().set_fit_request(sample_weight=True)
    >>> try:
    ...     GridSearchCV(
    ...         estimator=lr, param_grid=param_grid
    ...     ).fit(X, y, sample_weight=my_weights)
    ... except ValueError as e:
    ...     print(e)
    [sample_weight] are passed but are not explicitly set as requested or not
    requested for LogisticRegression.score, which is used within GridSearchCV.fit.
    Call `LogisticRegression.set_score_request({metadata}=True/False)` for each metadata
    you want to request/ignore. See the Metadata Routing User guide
    <https://scikit-learn.org/stable/metadata_routing.html> for more information.
    

The issue can be fixed by explicitly setting the request value:
    
    
    >>> lr = LogisticRegression().set_fit_request(
    ...     sample_weight=True
    ... ).set_score_request(sample_weight=False)
    

At the end of the **Usage Examples** section, we disable the configuration flag for metadata routing:
    
    
    >>> sklearn.set_config(enable_metadata_routing=False)
    

## 4.3. Metadata Routing Support Status

All consumers (i.e. simple estimators which only consume metadata and don’t route them) support metadata routing, meaning they can be used inside meta-estimators which support metadata routing. However, development of support for metadata routing for meta-estimators is in progress, and here is a list of meta-estimators and tools which support and don’t yet support metadata routing.

Meta-estimators and functions supporting metadata routing:

  * [`sklearn.calibration.CalibratedClassifierCV`](<modules/generated/sklearn.calibration.CalibratedClassifierCV.html#sklearn.calibration.CalibratedClassifierCV> "sklearn.calibration.CalibratedClassifierCV")

  * [`sklearn.compose.ColumnTransformer`](<modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer")

  * [`sklearn.compose.TransformedTargetRegressor`](<modules/generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor> "sklearn.compose.TransformedTargetRegressor")

  * [`sklearn.covariance.GraphicalLassoCV`](<modules/generated/sklearn.covariance.GraphicalLassoCV.html#sklearn.covariance.GraphicalLassoCV> "sklearn.covariance.GraphicalLassoCV")

  * [`sklearn.ensemble.StackingClassifier`](<modules/generated/sklearn.ensemble.StackingClassifier.html#sklearn.ensemble.StackingClassifier> "sklearn.ensemble.StackingClassifier")

  * [`sklearn.ensemble.StackingRegressor`](<modules/generated/sklearn.ensemble.StackingRegressor.html#sklearn.ensemble.StackingRegressor> "sklearn.ensemble.StackingRegressor")

  * [`sklearn.ensemble.VotingClassifier`](<modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier> "sklearn.ensemble.VotingClassifier")

  * [`sklearn.ensemble.VotingRegressor`](<modules/generated/sklearn.ensemble.VotingRegressor.html#sklearn.ensemble.VotingRegressor> "sklearn.ensemble.VotingRegressor")

  * [`sklearn.ensemble.BaggingClassifier`](<modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier> "sklearn.ensemble.BaggingClassifier")

  * [`sklearn.ensemble.BaggingRegressor`](<modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor> "sklearn.ensemble.BaggingRegressor")

  * [`sklearn.feature_selection.RFE`](<modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE> "sklearn.feature_selection.RFE")

  * [`sklearn.feature_selection.RFECV`](<modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV> "sklearn.feature_selection.RFECV")

  * [`sklearn.feature_selection.SelectFromModel`](<modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel> "sklearn.feature_selection.SelectFromModel")

  * [`sklearn.feature_selection.SequentialFeatureSelector`](<modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector> "sklearn.feature_selection.SequentialFeatureSelector")

  * [`sklearn.impute.IterativeImputer`](<modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer> "sklearn.impute.IterativeImputer")

  * [`sklearn.linear_model.ElasticNetCV`](<modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV> "sklearn.linear_model.ElasticNetCV")

  * [`sklearn.linear_model.LarsCV`](<modules/generated/sklearn.linear_model.LarsCV.html#sklearn.linear_model.LarsCV> "sklearn.linear_model.LarsCV")

  * [`sklearn.linear_model.LassoCV`](<modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV> "sklearn.linear_model.LassoCV")

  * [`sklearn.linear_model.LassoLarsCV`](<modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV> "sklearn.linear_model.LassoLarsCV")

  * [`sklearn.linear_model.LogisticRegressionCV`](<modules/generated/sklearn.linear_model.LogisticRegressionCV.html#sklearn.linear_model.LogisticRegressionCV> "sklearn.linear_model.LogisticRegressionCV")

  * [`sklearn.linear_model.MultiTaskElasticNetCV`](<modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html#sklearn.linear_model.MultiTaskElasticNetCV> "sklearn.linear_model.MultiTaskElasticNetCV")

  * [`sklearn.linear_model.MultiTaskLassoCV`](<modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV> "sklearn.linear_model.MultiTaskLassoCV")

  * [`sklearn.linear_model.OrthogonalMatchingPursuitCV`](<modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV> "sklearn.linear_model.OrthogonalMatchingPursuitCV")

  * [`sklearn.linear_model.RANSACRegressor`](<modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor> "sklearn.linear_model.RANSACRegressor")

  * [`sklearn.linear_model.RidgeClassifierCV`](<modules/generated/sklearn.linear_model.RidgeClassifierCV.html#sklearn.linear_model.RidgeClassifierCV> "sklearn.linear_model.RidgeClassifierCV")

  * [`sklearn.linear_model.RidgeCV`](<modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV> "sklearn.linear_model.RidgeCV")

  * [`sklearn.model_selection.GridSearchCV`](<modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV> "sklearn.model_selection.GridSearchCV")

  * [`sklearn.model_selection.HalvingGridSearchCV`](<modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV> "sklearn.model_selection.HalvingGridSearchCV")

  * [`sklearn.model_selection.HalvingRandomSearchCV`](<modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html#sklearn.model_selection.HalvingRandomSearchCV> "sklearn.model_selection.HalvingRandomSearchCV")

  * [`sklearn.model_selection.RandomizedSearchCV`](<modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV> "sklearn.model_selection.RandomizedSearchCV")

  * [`sklearn.model_selection.permutation_test_score`](<modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn.model_selection.permutation_test_score> "sklearn.model_selection.permutation_test_score")

  * [`sklearn.model_selection.cross_validate`](<modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate> "sklearn.model_selection.cross_validate")

  * [`sklearn.model_selection.cross_val_score`](<modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score> "sklearn.model_selection.cross_val_score")

  * [`sklearn.model_selection.cross_val_predict`](<modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict> "sklearn.model_selection.cross_val_predict")

  * [`sklearn.model_selection.learning_curve`](<modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve> "sklearn.model_selection.learning_curve")

  * [`sklearn.model_selection.validation_curve`](<modules/generated/sklearn.model_selection.validation_curve.html#sklearn.model_selection.validation_curve> "sklearn.model_selection.validation_curve")

  * [`sklearn.multiclass.OneVsOneClassifier`](<modules/generated/sklearn.multiclass.OneVsOneClassifier.html#sklearn.multiclass.OneVsOneClassifier> "sklearn.multiclass.OneVsOneClassifier")

  * [`sklearn.multiclass.OneVsRestClassifier`](<modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier> "sklearn.multiclass.OneVsRestClassifier")

  * [`sklearn.multiclass.OutputCodeClassifier`](<modules/generated/sklearn.multiclass.OutputCodeClassifier.html#sklearn.multiclass.OutputCodeClassifier> "sklearn.multiclass.OutputCodeClassifier")

  * [`sklearn.multioutput.ClassifierChain`](<modules/generated/sklearn.multioutput.ClassifierChain.html#sklearn.multioutput.ClassifierChain> "sklearn.multioutput.ClassifierChain")

  * [`sklearn.multioutput.MultiOutputClassifier`](<modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier> "sklearn.multioutput.MultiOutputClassifier")

  * [`sklearn.multioutput.MultiOutputRegressor`](<modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor> "sklearn.multioutput.MultiOutputRegressor")

  * [`sklearn.multioutput.RegressorChain`](<modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain> "sklearn.multioutput.RegressorChain")

  * [`sklearn.pipeline.FeatureUnion`](<modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion")

  * [`sklearn.pipeline.Pipeline`](<modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline")

  * [`sklearn.semi_supervised.SelfTrainingClassifier`](<modules/generated/sklearn.semi_supervised.SelfTrainingClassifier.html#sklearn.semi_supervised.SelfTrainingClassifier> "sklearn.semi_supervised.SelfTrainingClassifier")




Meta-estimators and tools not supporting metadata routing yet:

  * [`sklearn.ensemble.AdaBoostClassifier`](<modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier> "sklearn.ensemble.AdaBoostClassifier")

  * [`sklearn.ensemble.AdaBoostRegressor`](<modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor> "sklearn.ensemble.AdaBoostRegressor")



