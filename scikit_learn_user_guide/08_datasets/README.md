# 8\. Dataset loading utilities

The `sklearn.datasets` package embeds some small toy datasets and provides helpers to fetch larger datasets commonly used by the machine learning community to benchmark algorithms on data that comes from the ‘real world’.

To evaluate the impact of the scale of the dataset (`n_samples` and `n_features`) while controlling the statistical properties of the data (typically the correlation and informativeness of the features), it is also possible to generate synthetic data.

**General dataset API.** There are three main kinds of dataset interfaces that can be used to get datasets depending on the desired type of dataset.

**The dataset loaders.** They can be used to load small standard datasets, described in the [Toy datasets](<datasets/toy_dataset.html#toy-datasets>) section.

**The dataset fetchers.** They can be used to download and load larger datasets, described in the [Real world datasets](<datasets/real_world.html#real-world-datasets>) section.

Both loaders and fetchers functions return a [`Bunch`](<modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch> "sklearn.utils.Bunch") object holding at least two items: an array of shape `n_samples` * `n_features` with key `data` (except for 20newsgroups) and a numpy array of length `n_samples`, containing the target values, with key `target`.

The Bunch object is a dictionary that exposes its keys as attributes. For more information about Bunch object, see [`Bunch`](<modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch> "sklearn.utils.Bunch").

It’s also possible for almost all of these functions to constrain the output to be a tuple containing only the data and the target, by setting the `return_X_y` parameter to `True`.

The datasets also contain a full description in their `DESCR` attribute and some contain `feature_names` and `target_names`. See the dataset descriptions below for details.

**The dataset generation functions.** They can be used to generate controlled synthetic datasets, described in the [Generated datasets](<datasets/sample_generators.html#sample-generators>) section.

These functions return a tuple `(X, y)` consisting of a `n_samples` * `n_features` numpy array `X` and an array of length `n_samples` containing the targets `y`.

In addition, there are also miscellaneous tools to load datasets of other formats or from other locations, described in the [Loading other datasets](<datasets/loading_other_datasets.html#loading-other-datasets>) section.

  * [8.1. Toy datasets](<datasets/toy_dataset.html>)
    * [8.1.1. Iris plants dataset](<datasets/toy_dataset.html#iris-plants-dataset>)
    * [8.1.2. Diabetes dataset](<datasets/toy_dataset.html#diabetes-dataset>)
    * [8.1.3. Optical recognition of handwritten digits dataset](<datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset>)
    * [8.1.4. Linnerrud dataset](<datasets/toy_dataset.html#linnerrud-dataset>)
    * [8.1.5. Wine recognition dataset](<datasets/toy_dataset.html#wine-recognition-dataset>)
    * [8.1.6. Breast cancer Wisconsin (diagnostic) dataset](<datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset>)
  * [8.2. Real world datasets](<datasets/real_world.html>)
    * [8.2.1. The Olivetti faces dataset](<datasets/real_world.html#the-olivetti-faces-dataset>)
    * [8.2.2. The 20 newsgroups text dataset](<datasets/real_world.html#the-20-newsgroups-text-dataset>)
    * [8.2.3. The Labeled Faces in the Wild face recognition dataset](<datasets/real_world.html#the-labeled-faces-in-the-wild-face-recognition-dataset>)
    * [8.2.4. Forest covertypes](<datasets/real_world.html#forest-covertypes>)
    * [8.2.5. RCV1 dataset](<datasets/real_world.html#rcv1-dataset>)
    * [8.2.6. Kddcup 99 dataset](<datasets/real_world.html#kddcup-99-dataset>)
    * [8.2.7. California Housing dataset](<datasets/real_world.html#california-housing-dataset>)
    * [8.2.8. Species distribution dataset](<datasets/real_world.html#species-distribution-dataset>)
  * [8.3. Generated datasets](<datasets/sample_generators.html>)
    * [8.3.1. Generators for classification and clustering](<datasets/sample_generators.html#generators-for-classification-and-clustering>)
    * [8.3.2. Generators for regression](<datasets/sample_generators.html#generators-for-regression>)
    * [8.3.3. Generators for manifold learning](<datasets/sample_generators.html#generators-for-manifold-learning>)
    * [8.3.4. Generators for decomposition](<datasets/sample_generators.html#generators-for-decomposition>)
  * [8.4. Loading other datasets](<datasets/loading_other_datasets.html>)
    * [8.4.1. Sample images](<datasets/loading_other_datasets.html#sample-images>)
    * [8.4.2. Datasets in svmlight / libsvm format](<datasets/loading_other_datasets.html#datasets-in-svmlight-libsvm-format>)
    * [8.4.3. Downloading datasets from the openml.org repository](<datasets/loading_other_datasets.html#downloading-datasets-from-the-openml-org-repository>)
    * [8.4.4. Loading from external datasets](<datasets/loading_other_datasets.html#loading-from-external-datasets>)


