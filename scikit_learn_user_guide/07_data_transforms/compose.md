# 7.1. Pipelines and composite estimators

To build a composite estimator, transformers are usually combined with other transformers or with [predictors](<../glossary.html#term-predictors>) (such as classifiers or regressors). The most common tool used for composing estimators is a Pipeline. Pipelines require all steps except the last to be a [transformer](<../glossary.html#term-transformer>). The last step can be anything, a transformer, a [predictor](<../glossary.html#term-predictor>), or a clustering estimator which might have or not have a `.predict(...)` method. A pipeline exposes all methods provided by the last estimator: if the last step provides a `transform` method, then the pipeline would have a `transform` method and behave like a transformer. If the last step provides a `predict` method, then the pipeline would expose that method, and given a data [X](<../glossary.html#term-X>), use all steps except the last to transform the data, and then give that transformed data to the `predict` method of the last step of the pipeline. The class `Pipeline` is often used in combination with ColumnTransformer or FeatureUnion which concatenate the output of transformers into a composite feature space. TransformedTargetRegressor deals with transforming the [target](<../glossary.html#term-target>) (i.e. log-transform [y](<../glossary.html#term-y>)).

## 7.1.1. Pipeline: chaining estimators

[`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") can be used to chain multiple estimators into one. This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection, normalization and classification. [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") serves multiple purposes here:

Convenience and encapsulation
    

You only have to call [fit](<../glossary.html#term-fit>) and [predict](<../glossary.html#term-predict>) once on your data to fit a whole sequence of estimators.

Joint parameter selection
    

You can [grid search](<grid_search.html#grid-search>) over parameters of all estimators in the pipeline at once.

Safety
    

Pipelines help avoid leaking statistics from your test data into the trained model in cross-validation, by ensuring that the same samples are used to train the transformers and predictors.

All estimators in a pipeline, except the last one, must be transformers (i.e. must have a [transform](<../glossary.html#term-transform>) method). The last estimator may be any type (transformer, classifier, etc.).

Note

Calling `fit` on the pipeline is the same as calling `fit` on each estimator in turn, `transform` the input and pass it on to the next step. The pipeline has all the methods that the last estimator in the pipeline has, i.e. if the last estimator is a classifier, the [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") can be used as a classifier. If the last estimator is a transformer, again, so is the pipeline.

### 7.1.1.1. Usage

#### 7.1.1.1.1. Build a pipeline

The [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") is built using a list of `(key, value)` pairs, where the `key` is a string containing the name you want to give this step and `value` is an estimator object:
    
    
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> from sklearn.decomposition import PCA
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> pipe = Pipeline(estimators)
    >>> pipe
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC())])
    

Shorthand version using [`make_pipeline`](<generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline> "sklearn.pipeline.make_pipeline")

The utility function [`make_pipeline`](<generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline> "sklearn.pipeline.make_pipeline") is a shorthand for constructing pipelines; it takes a variable number of estimators and returns a pipeline, filling in the names automatically:
    
    
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(PCA(), SVC())
    Pipeline(steps=[('pca', PCA()), ('svc', SVC())])
    

#### 7.1.1.1.2. Access pipeline steps

The estimators of a pipeline are stored as a list in the `steps` attribute. A sub-pipeline can be extracted using the slicing notation commonly used for Python Sequences such as lists or strings (although only a step of 1 is permitted). This is convenient for performing only some of the transformations (or their inverse):
    
    
    >>> pipe[:1]
    Pipeline(steps=[('reduce_dim', PCA())])
    >>> pipe[-1:]
    Pipeline(steps=[('clf', SVC())])
    

Accessing a step by name or position

A specific step can also be accessed by index or name by indexing (with `[idx]`) the pipeline:
    
    
    >>> pipe.steps[0]
    ('reduce_dim', PCA())
    >>> pipe[0]
    PCA()
    >>> pipe['reduce_dim']
    PCA()
    

`Pipeline`’s `named_steps` attribute allows accessing steps by name with tab completion in interactive environments:
    
    
    >>> pipe.named_steps.reduce_dim is pipe['reduce_dim']
    True
    

#### 7.1.1.1.3. Tracking feature names in a pipeline

To enable model inspection, [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") has a `get_feature_names_out()` method, just like all transformers. You can use pipeline slicing to get the feature names going into each step:
    
    
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_selection import SelectKBest
    >>> iris = load_iris()
    >>> pipe = Pipeline(steps=[
    ...    ('select', SelectKBest(k=2)),
    ...    ('clf', LogisticRegression())])
    >>> pipe.fit(iris.data, iris.target)
    Pipeline(steps=[('select', SelectKBest(...)), ('clf', LogisticRegression(...))])
    >>> pipe[:-1].get_feature_names_out()
    array(['x2', 'x3'], ...)
    

Customize feature names

You can also provide custom feature names for the input data using `get_feature_names_out`:
    
    
    >>> pipe[:-1].get_feature_names_out(iris.feature_names)
    array(['petal length (cm)', 'petal width (cm)'], ...)
    

#### 7.1.1.1.4. Access to nested parameters

It is common to adjust the parameters of an estimator within a pipeline. This parameter is therefore nested because it belongs to a particular sub-step. Parameters of the estimators in the pipeline are accessible using the `<estimator>__<parameter>` syntax:
    
    
    >>> pipe = Pipeline(steps=[("reduce_dim", PCA()), ("clf", SVC())])
    >>> pipe.set_params(clf__C=10)
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC(C=10))])
    

When does it matter?

This is particularly important for doing grid searches:
    
    
    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = dict(reduce_dim__n_components=[2, 5, 10],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)
    

Individual steps may also be replaced as parameters, and non-final steps may be ignored by setting them to `'passthrough'`:
    
    
    >>> param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
    ...                   clf=[SVC(), LogisticRegression()],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)
    

See also

  * [Composite estimators and parameter spaces](<grid_search.html#composite-grid-search>)




Examples

  * [Pipeline ANOVA SVM](<../auto_examples/feature_selection/plot_feature_selection_pipeline.html#sphx-glr-auto-examples-feature-selection-plot-feature-selection-pipeline-py>)

  * [Sample pipeline for text feature extraction and evaluation](<../auto_examples/model_selection/plot_grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-plot-grid-search-text-feature-extraction-py>)

  * [Pipelining: chaining a PCA and a logistic regression](<../auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py>)

  * [Explicit feature map approximation for RBF kernels](<../auto_examples/miscellaneous/plot_kernel_approximation.html#sphx-glr-auto-examples-miscellaneous-plot-kernel-approximation-py>)

  * [SVM-Anova: SVM with univariate feature selection](<../auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py>)

  * [Selecting dimensionality reduction with Pipeline and GridSearchCV](<../auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py>)

  * [Displaying Pipelines](<../auto_examples/miscellaneous/plot_pipeline_display.html#sphx-glr-auto-examples-miscellaneous-plot-pipeline-display-py>)




### 7.1.1.2. Caching transformers: avoid repeated computation

Fitting transformers may be computationally expensive. With its `memory` parameter set, [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") will cache each transformer after calling `fit`. This feature is used to avoid computing the fit transformers within a pipeline if the parameters and input data are identical. A typical example is the case of a grid search in which the transformers can be fitted only once and reused for each configuration. The last step will never be cached, even if it is a transformer.

The parameter `memory` is needed in order to cache the transformers. `memory` can be either a string containing the directory where to cache the transformers or a [joblib.Memory](<https://joblib.readthedocs.io/en/latest/memory.html>) object:
    
    
    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from sklearn.pipeline import Pipeline
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> cachedir = mkdtemp()
    >>> pipe = Pipeline(estimators, memory=cachedir)
    >>> pipe
    Pipeline(memory=...,
             steps=[('reduce_dim', PCA()), ('clf', SVC())])
    >>> # Clear the cache directory when you don't need it anymore
    >>> rmtree(cachedir)
    

Side effect of caching transformers

Using a [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") without cache enabled, it is possible to inspect the original instance such as:
    
    
    >>> from sklearn.datasets import load_digits
    >>> X_digits, y_digits = load_digits(return_X_y=True)
    >>> pca1 = PCA(n_components=10)
    >>> svm1 = SVC()
    >>> pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
    >>> pipe.fit(X_digits, y_digits)
    Pipeline(steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
    >>> # The pca instance can be inspected directly
    >>> pca1.components_.shape
    (10, 64)
    

Enabling caching triggers a clone of the transformers before fitting. Therefore, the transformer instance given to the pipeline cannot be inspected directly. In the following example, accessing the [`PCA`](<generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA> "sklearn.decomposition.PCA") instance `pca2` will raise an `AttributeError` since `pca2` will be an unfitted transformer. Instead, use the attribute `named_steps` to inspect estimators within the pipeline:
    
    
    >>> cachedir = mkdtemp()
    >>> pca2 = PCA(n_components=10)
    >>> svm2 = SVC()
    >>> cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
    ...                        memory=cachedir)
    >>> cached_pipe.fit(X_digits, y_digits)
    Pipeline(memory=...,
             steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
    >>> cached_pipe.named_steps['reduce_dim'].components_.shape
    (10, 64)
    >>> # Remove the cache directory
    >>> rmtree(cachedir)
    

Examples

  * [Selecting dimensionality reduction with Pipeline and GridSearchCV](<../auto_examples/compose/plot_compare_reduction.html#sphx-glr-auto-examples-compose-plot-compare-reduction-py>)




## 7.1.2. Transforming target in regression

[`TransformedTargetRegressor`](<generated/sklearn.compose.TransformedTargetRegressor.html#sklearn.compose.TransformedTargetRegressor> "sklearn.compose.TransformedTargetRegressor") transforms the targets `y` before fitting a regression model. The predictions are mapped back to the original space via an inverse transform. It takes as an argument the regressor that will be used for prediction, and the transformer that will be applied to the target variable:
    
    
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.preprocessing import QuantileTransformer
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> # create a synthetic dataset
    >>> X, y = make_regression(n_samples=20640,
    ...                        n_features=8,
    ...                        noise=100.0,
    ...                        random_state=0)
    >>> y = np.exp( 1 + (y - y.min()) * (4 / (y.max() - y.min())))
    >>> X, y = X[:2000, :], y[:2000]  # select a subset of data
    >>> transformer = QuantileTransformer(output_distribution='normal')
    >>> regressor = LinearRegression()
    >>> regr = TransformedTargetRegressor(regressor=regressor,
    ...                                   transformer=transformer)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    >>> regr.fit(X_train, y_train)
    TransformedTargetRegressor(...)
    >>> print(f"R2 score: {regr.score(X_test, y_test):.2f}")
    R2 score: 0.67
    >>> raw_target_regr = LinearRegression().fit(X_train, y_train)
    >>> print(f"R2 score: {raw_target_regr.score(X_test, y_test):.2f}")
    R2 score: 0.64
    

For simple transformations, instead of a Transformer object, a pair of functions can be passed, defining the transformation and its inverse mapping:
    
    
    >>> def func(x):
    ...     return np.log(x)
    >>> def inverse_func(x):
    ...     return np.exp(x)
    

Subsequently, the object is created as:
    
    
    >>> regr = TransformedTargetRegressor(regressor=regressor,
    ...                                   func=func,
    ...                                   inverse_func=inverse_func)
    >>> regr.fit(X_train, y_train)
    TransformedTargetRegressor(...)
    >>> print(f"R2 score: {regr.score(X_test, y_test):.2f}")
    R2 score: 0.67
    

By default, the provided functions are checked at each fit to be the inverse of each other. However, it is possible to bypass this checking by setting `check_inverse` to `False`:
    
    
    >>> def inverse_func(x):
    ...     return x
    >>> regr = TransformedTargetRegressor(regressor=regressor,
    ...                                   func=func,
    ...                                   inverse_func=inverse_func,
    ...                                   check_inverse=False)
    >>> regr.fit(X_train, y_train)
    TransformedTargetRegressor(...)
    >>> print(f"R2 score: {regr.score(X_test, y_test):.2f}")
    R2 score: -3.02
    

Note

The transformation can be triggered by setting either `transformer` or the pair of functions `func` and `inverse_func`. However, setting both options will raise an error.

Examples

  * [Effect of transforming the targets in regression model](<../auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py>)




## 7.1.3. FeatureUnion: composite feature spaces

[`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") combines several transformer objects into a new transformer that combines their output. A [`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") takes a list of transformer objects. During fitting, each of these is fit to the data independently. The transformers are applied in parallel, and the feature matrices they output are concatenated side-by-side into a larger matrix.

When you want to apply different transformations to each field of the data, see the related class [`ColumnTransformer`](<generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer") (see user guide).

[`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") serves the same purposes as [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") \- convenience and joint parameter estimation and validation.

[`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") and [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") can be combined to create complex models.

(A [`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") has no way of checking whether two transformers might produce identical features. It only produces a union when the feature sets are disjoint, and making sure they are is the caller’s responsibility.)

### 7.1.3.1. Usage

A [`FeatureUnion`](<generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion> "sklearn.pipeline.FeatureUnion") is built using a list of `(key, value)` pairs, where the `key` is the name you want to give to a given transformation (an arbitrary string; it only serves as an identifier) and `value` is an estimator object:
    
    
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.decomposition import KernelPCA
    >>> estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
    >>> combined = FeatureUnion(estimators)
    >>> combined
    FeatureUnion(transformer_list=[('linear_pca', PCA()),
                                   ('kernel_pca', KernelPCA())])
    

Like pipelines, feature unions have a shorthand constructor called [`make_union`](<generated/sklearn.pipeline.make_union.html#sklearn.pipeline.make_union> "sklearn.pipeline.make_union") that does not require explicit naming of the components.

Like `Pipeline`, individual steps may be replaced using `set_params`, and ignored by setting to `'drop'`:
    
    
    >>> combined.set_params(kernel_pca='drop')
    FeatureUnion(transformer_list=[('linear_pca', PCA()),
                                   ('kernel_pca', 'drop')])
    

Examples

  * [Concatenating multiple feature extraction methods](<../auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py>)




## 7.1.4. ColumnTransformer for heterogeneous data

Many datasets contain features of different types, say text, floats, and dates, where each type of feature requires separate preprocessing or feature extraction steps. Often it is easiest to preprocess data before applying scikit-learn methods, for example using [pandas](<https://pandas.pydata.org/>). Processing your data before passing it to scikit-learn might be problematic for one of the following reasons:

  1. Incorporating statistics from test data into the preprocessors makes cross-validation scores unreliable (known as _data leakage_), for example in the case of scalers or imputing missing values.

  2. You may want to include the parameters of the preprocessors in a [parameter search](<grid_search.html#grid-search>).




The [`ColumnTransformer`](<generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer") helps performing different transformations for different columns of the data, within a [`Pipeline`](<generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline> "sklearn.pipeline.Pipeline") that is safe from data leakage and that can be parametrized. [`ColumnTransformer`](<generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer") works on arrays, sparse matrices, and [pandas DataFrames](<https://pandas.pydata.org/pandas-docs/stable/>).

To each column, a different transformation can be applied, such as preprocessing or a specific feature extraction method:
    
    
    >>> import pandas as pd
    >>> X = pd.DataFrame(
    ...     {'city': ['London', 'London', 'Paris', 'Sallisaw'],
    ...      'title': ["His Last Bow", "How Watson Learned the Trick",
    ...                "A Moveable Feast", "The Grapes of Wrath"],
    ...      'expert_rating': [5, 3, 4, 5],
    ...      'user_rating': [4, 5, 4, 3]})
    

For this data, we might want to encode the `'city'` column as a categorical variable using [`OneHotEncoder`](<generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder> "sklearn.preprocessing.OneHotEncoder") but apply a [`CountVectorizer`](<generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer> "sklearn.feature_extraction.text.CountVectorizer") to the `'title'` column. As we might use multiple feature extraction methods on the same column, we give each transformer a unique name, say `'city_category'` and `'title_bow'`. By default, the remaining rating columns are ignored (`remainder='drop'`):
    
    
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> column_trans = ColumnTransformer(
    ...     [('categories', OneHotEncoder(dtype='int'), ['city']),
    ...      ('title_bow', CountVectorizer(), 'title')],
    ...     remainder='drop', verbose_feature_names_out=False)
    
    >>> column_trans.fit(X)
    ColumnTransformer(transformers=[('categories', OneHotEncoder(dtype='int'),
                                     ['city']),
                                    ('title_bow', CountVectorizer(), 'title')],
                      verbose_feature_names_out=False)
    
    >>> column_trans.get_feature_names_out()
    array(['city_London', 'city_Paris', 'city_Sallisaw', 'bow', 'feast',
    'grapes', 'his', 'how', 'last', 'learned', 'moveable', 'of', 'the',
     'trick', 'watson', 'wrath'], ...)
    
    >>> column_trans.transform(X).toarray()
    array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]]...)
    

In the above example, the [`CountVectorizer`](<generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer> "sklearn.feature_extraction.text.CountVectorizer") expects a 1D array as input and therefore the columns were specified as a string (`'title'`). However, [`OneHotEncoder`](<generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder> "sklearn.preprocessing.OneHotEncoder") as most of other transformers expects 2D data, therefore in that case you need to specify the column as a list of strings (`['city']`).

Apart from a scalar or a single item list, the column selection can be specified as a list of multiple items, an integer array, a slice, a boolean mask, or with a [`make_column_selector`](<generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector> "sklearn.compose.make_column_selector"). The [`make_column_selector`](<generated/sklearn.compose.make_column_selector.html#sklearn.compose.make_column_selector> "sklearn.compose.make_column_selector") is used to select columns based on data type or column name:
    
    
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.compose import make_column_selector
    >>> ct = ColumnTransformer([
    ...       ('scale', StandardScaler(),
    ...       make_column_selector(dtype_include=np.number)),
    ...       ('onehot',
    ...       OneHotEncoder(),
    ...       make_column_selector(pattern='city', dtype_include=object))])
    >>> ct.fit_transform(X)
    array([[ 0.904,  0.      ,  1. ,  0. ,  0. ],
           [-1.507,  1.414,  1. ,  0. ,  0. ],
           [-0.301,  0.      ,  0. ,  1. ,  0. ],
           [ 0.904, -1.414,  0. ,  0. ,  1. ]])
    

Strings can reference columns if the input is a DataFrame, integers are always interpreted as the positional columns.

We can keep the remaining rating columns by setting `remainder='passthrough'`. The values are appended to the end of the transformation:
    
    
    >>> column_trans = ColumnTransformer(
    ...     [('city_category', OneHotEncoder(dtype='int'),['city']),
    ...      ('title_bow', CountVectorizer(), 'title')],
    ...     remainder='passthrough')
    
    >>> column_trans.fit_transform(X)
    array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4],
           [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 3, 5],
           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 4],
           [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 5, 3]]...)
    

The `remainder` parameter can be set to an estimator to transform the remaining rating columns. The transformed values are appended to the end of the transformation:
    
    
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> column_trans = ColumnTransformer(
    ...     [('city_category', OneHotEncoder(), ['city']),
    ...      ('title_bow', CountVectorizer(), 'title')],
    ...     remainder=MinMaxScaler())
    
    >>> column_trans.fit_transform(X)[:, -2:]
    array([[1. , 0.5],
           [0. , 1. ],
           [0.5, 0.5],
           [1. , 0. ]])
    

The [`make_column_transformer`](<generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer> "sklearn.compose.make_column_transformer") function is available to more easily create a [`ColumnTransformer`](<generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer") object. Specifically, the names will be given automatically. The equivalent for the above example would be:
    
    
    >>> from sklearn.compose import make_column_transformer
    >>> column_trans = make_column_transformer(
    ...     (OneHotEncoder(), ['city']),
    ...     (CountVectorizer(), 'title'),
    ...     remainder=MinMaxScaler())
    >>> column_trans
    ColumnTransformer(remainder=MinMaxScaler(),
                      transformers=[('onehotencoder', OneHotEncoder(), ['city']),
                                    ('countvectorizer', CountVectorizer(),
                                     'title')])
    

If [`ColumnTransformer`](<generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer> "sklearn.compose.ColumnTransformer") is fitted with a dataframe and the dataframe only has string column names, then transforming a dataframe will use the column names to select the columns:
    
    
    >>> ct = ColumnTransformer(
    ...          [("scale", StandardScaler(), ["expert_rating"])]).fit(X)
    >>> X_new = pd.DataFrame({"expert_rating": [5, 6, 1],
    ...                       "ignored_new_col": [1.2, 0.3, -0.1]})
    >>> ct.transform(X_new)
    array([[ 0.9],
           [ 2.1],
           [-3.9]])
    

## 7.1.5. Visualizing Composite Estimators

Estimators are displayed with an HTML representation when shown in a jupyter notebook. This is useful to diagnose or visualize a Pipeline with many estimators. This visualization is activated by default:
    
    
    >>> column_trans  
    

It can be deactivated by setting the `display` option in [`set_config`](<generated/sklearn.set_config.html#sklearn.set_config> "sklearn.set_config") to ‘text’:
    
    
    >>> from sklearn import set_config
    >>> set_config(display='text')  
    >>> # displays text representation in a jupyter context
    >>> column_trans  
    

An example of the HTML output can be seen in the **HTML representation of Pipeline** section of [Column Transformer with Mixed Types](<../auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py>). As an alternative, the HTML can be written to a file using [`estimator_html_repr`](<generated/sklearn.utils.estimator_html_repr.html#sklearn.utils.estimator_html_repr> "sklearn.utils.estimator_html_repr"):
    
    
    >>> from sklearn.utils import estimator_html_repr
    >>> with open('my_estimator.html', 'w') as f:  
    ...     f.write(estimator_html_repr(clf))
    

Examples

  * [Column Transformer with Heterogeneous Data Sources](<../auto_examples/compose/plot_column_transformer.html#sphx-glr-auto-examples-compose-plot-column-transformer-py>)

  * [Column Transformer with Mixed Types](<../auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py>)



