# 10\. Model persistence

Summary of model persistence methods Persistence method | Pros | Risks / Cons  
---|---|---  
ONNX | 

  * Serve models without a Python environment
  * Serving and training environments independent of one another
  * Most secure option

| 

  * Not all scikit-learn models are supported
  * Custom estimators require more work to support
  * Original Python object is lost and cannot be reconstructed

  
skops.io | 

  * More secure than `pickle` based formats
  * Contents can be partly validated without loading

| 

  * Not as fast as `pickle` based formats
  * Supports less types than `pickle` based formats
  * Requires the same environment as the training environment

  
[`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") | 

  * Native to Python
  * Can serialize most Python objects
  * Efficient memory usage with `protocol=5`

| 

  * Loading can execute arbitrary code
  * Requires the same environment as the training environment

  
[`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)") | 

  * Efficient memory usage
  * Supports memory mapping
  * Easy shortcuts for compression and decompression

| 

  * Pickle based format
  * Loading can execute arbitrary code
  * Requires the same environment as the training environment

  
[cloudpickle](<https://github.com/cloudpipe/cloudpickle>) | 

  * Can serialize non-packaged, custom Python code
  * Comparable loading efficiency as [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") with `protocol=5`

| 

  * Pickle based format
  * Loading can execute arbitrary code
  * No forward compatibility guarantees
  * Requires the same environment as the training environment

  
  
After training a scikit-learn model, it is desirable to have a way to persist the model for future use without having to retrain. Based on your use-case, there are a few different ways to persist a scikit-learn model, and here we help you decide which one suits you best. In order to make a decision, you need to answer the following questions:

  1. Do you need the Python object after persistence, or do you only need to persist in order to serve the model and get predictions out of it?




If you only need to serve the model and no further investigation on the Python object itself is required, then ONNX might be the best fit for you. Note that not all models are supported by ONNX.

In case ONNX is not suitable for your use-case, the next question is:

  2. Do you absolutely trust the source of the model, or are there any security concerns regarding where the persisted model comes from?




If you have security concerns, then you should consider using skops.io which gives you back the Python object, but unlike `pickle` based persistence solutions, loading the persisted model doesn’t automatically allow arbitrary code execution. Note that this requires manual investigation of the persisted file, which [`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)") allows you to do.

The other solutions assume you absolutely trust the source of the file to be loaded, as they are all susceptible to arbitrary code execution upon loading the persisted file since they all use the pickle protocol under the hood.

  3. Do you care about the performance of loading the model, and sharing it between processes where a memory mapped object on disk is beneficial?




If yes, then you can consider using joblib. If this is not a major concern for you, then you can use the built-in [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") module.

  4. Did you try [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") or [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)") and found that the model cannot be persisted? It can happen for instance when you have user defined functions in your model.




If yes, then you can use [cloudpickle](<https://github.com/cloudpipe/cloudpickle>) which can serialize certain objects which cannot be serialized by [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") or [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)").

## 10.1. Workflow Overview

In a typical workflow, the first step is to train the model using scikit-learn and scikit-learn compatible libraries. Note that support for scikit-learn and third party estimators varies across the different persistence methods.

### 10.1.1. Train and Persist the Model

Creating an appropriate model depends on your use-case. As an example, here we train a [`sklearn.ensemble.HistGradientBoostingClassifier`](<modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier> "sklearn.ensemble.HistGradientBoostingClassifier") on the iris dataset:
    
    
    >>> from sklearn import ensemble
    >>> from sklearn import datasets
    >>> clf = ensemble.HistGradientBoostingClassifier()
    >>> X, y = datasets.load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    HistGradientBoostingClassifier()
    

Once the model is trained, you can persist it using your desired method, and then you can load the model in a separate environment and get predictions from it given input data. Here there are two major paths depending on how you persist and plan to serve the model:

  * ONNX: You need an `ONNX` runtime and an environment with appropriate dependencies installed to load the model and use the runtime to get predictions. This environment can be minimal and does not necessarily even require Python to be installed to load the model and compute predictions. Also note that `onnxruntime` typically requires much less RAM than Python to compute predictions from small models.

  * [`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)"), [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"), [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)"), [cloudpickle](<https://github.com/cloudpipe/cloudpickle>): You need a Python environment with the appropriate dependencies installed to load the model and get predictions from it. This environment should have the same **packages** and the same **versions** as the environment where the model was trained. Note that none of these methods support loading a model trained with a different version of scikit-learn, and possibly different versions of other dependencies such as `numpy` and `scipy`. Another concern would be running the persisted model on a different hardware, and in most cases you should be able to load your persisted model on a different hardware.




## 10.2. ONNX

`ONNX`, or [Open Neural Network Exchange](<https://onnx.ai/>) format is best suitable in use-cases where one needs to persist the model and then use the persisted artifact to get predictions without the need to load the Python object itself. It is also useful in cases where the serving environment needs to be lean and minimal, since the `ONNX` runtime does not require `python`.

`ONNX` is a binary serialization of the model. It has been developed to improve the usability of the interoperable representation of data models. It aims to facilitate the conversion of the data models between different machine learning frameworks, and to improve their portability on different computing architectures. More details are available from the [ONNX tutorial](<https://onnx.ai/get-started.html>). To convert scikit-learn model to `ONNX` [sklearn-onnx](<http://onnx.ai/sklearn-onnx/>) has been developed. However, not all scikit-learn models are supported, and it is limited to the core scikit-learn and does not support most third party estimators. One can write a custom converter for third party or custom estimators, but the documentation to do that is sparse and it might be challenging to do so.

Using ONNX

To convert the model to `ONNX` format, you need to give the converter some information about the input as well, about which you can read more [here](<http://onnx.ai/sklearn-onnx/index.html>):
    
    
    from skl2onnx import to_onnx
    onx = to_onnx(clf, X[:1].astype(numpy.float32), target_opset=12)
    with open("filename.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    

You can load the model in Python and use the `ONNX` runtime to get predictions:
    
    
    from onnxruntime import InferenceSession
    with open("filename.onnx", "rb") as f:
        onx = f.read()
    sess = InferenceSession(onx, providers=["CPUExecutionProvider"])
    pred_ort = sess.run(None, {"X": X_test.astype(numpy.float32)})[0]
    

## 10.3. `skops.io`

[`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)") avoids using [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") and only loads files which have types and references to functions which are trusted either by default or by the user. Therefore it provides a more secure format than [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"), [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)"), and [cloudpickle](<https://github.com/cloudpipe/cloudpickle>).

Using skops

The API is very similar to [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"), and you can persist your models as explained in the [documentation](<https://skops.readthedocs.io/en/stable/persistence.html>) using [`skops.io.dump`](<https://skops.readthedocs.io/en/stable/modules/classes.html#skops.io.dump> "\(in skops\)") and [`skops.io.dumps`](<https://skops.readthedocs.io/en/stable/modules/classes.html#skops.io.dumps> "\(in skops\)"):
    
    
    import skops.io as sio
    obj = sio.dump(clf, "filename.skops")
    

And you can load them back using [`skops.io.load`](<https://skops.readthedocs.io/en/stable/modules/classes.html#skops.io.load> "\(in skops\)") and [`skops.io.loads`](<https://skops.readthedocs.io/en/stable/modules/classes.html#skops.io.loads> "\(in skops\)"). However, you need to specify the types which are trusted by you. You can get existing unknown types in a dumped object / file using [`skops.io.get_untrusted_types`](<https://skops.readthedocs.io/en/stable/modules/classes.html#skops.io.get_untrusted_types> "\(in skops\)"), and after checking its contents, pass it to the load function:
    
    
    unknown_types = sio.get_untrusted_types(file="filename.skops")
    # investigate the contents of unknown_types, and only load if you trust
    # everything you see.
    clf = sio.load("filename.skops", trusted=unknown_types)
    

Please report issues and feature requests related to this format on the [skops issue tracker](<https://github.com/skops-dev/skops/issues>).

## 10.4. `pickle`, `joblib`, and `cloudpickle`

These three modules / packages, use the `pickle` protocol under the hood, but come with slight variations:

  * [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") is a module from the Python Standard Library. It can serialize and deserialize any Python object, including custom Python classes and objects.

  * [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)") is more efficient than `pickle` when working with large machine learning models or large numpy arrays.

  * [cloudpickle](<https://github.com/cloudpipe/cloudpickle>) can serialize certain objects which cannot be serialized by [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") or [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)"), such as user defined functions and lambda functions. This can happen for instance, when using a [`FunctionTransformer`](<modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer> "sklearn.preprocessing.FunctionTransformer") and using a custom function to transform the data.


Using `pickle`, `joblib`, or `cloudpickle`

Depending on your use-case, you can choose one of these three methods to persist and load your scikit-learn model, and they all follow the same API:
    
    
    # Here you can replace pickle with joblib or cloudpickle
    from pickle import dump
    with open("filename.pkl", "wb") as f:
        dump(clf, f, protocol=5)
    

Using `protocol=5` is recommended to reduce memory usage and make it faster to store and load any large NumPy array stored as a fitted attribute in the model. You can alternatively pass `protocol=pickle.HIGHEST_PROTOCOL` which is equivalent to `protocol=5` in Python 3.8 and later (at the time of writing).

And later when needed, you can load the same object from the persisted file:
    
    
    # Here you can replace pickle with joblib or cloudpickle
    from pickle import load
    with open("filename.pkl", "rb") as f:
        clf = load(f)
    

## 10.5. Security & Maintainability Limitations

[`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") (and [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)") and `cloudpickle` by extension), has many documented security vulnerabilities by design and should only be used if the artifact, i.e. the pickle-file, is coming from a trusted and verified source. You should never load a pickle file from an untrusted source, similarly to how you should never execute code from an untrusted source.

Also note that arbitrary computations can be represented using the `ONNX` format, and it is therefore recommended to serve models using `ONNX` in a sandboxed environment to safeguard against computational and memory exploits.

Also note that there are no supported ways to load a model trained with a different version of scikit-learn. While using [`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)"), [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)"), [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"), or [cloudpickle](<https://github.com/cloudpipe/cloudpickle>), models saved using one version of scikit-learn might load in other versions, however, this is entirely unsupported and inadvisable. It should also be kept in mind that operations performed on such data could give different and unexpected results, or even crash your Python process.

In order to rebuild a similar model with future versions of scikit-learn, additional metadata should be saved along the pickled model:

  * The training data, e.g. a reference to an immutable snapshot

  * The Python source code used to generate the model

  * The versions of scikit-learn and its dependencies

  * The cross validation score obtained on the training data




This should make it possible to check that the cross-validation score is in the same range as before.

Aside for a few exceptions, persisted models should be portable across operating systems and hardware architectures assuming the same versions of dependencies and Python are used. If you encounter an estimator that is not portable, please open an issue on GitHub. Persisted models are often deployed in production using containers like Docker, in order to freeze the environment and dependencies.

If you want to know more about these issues, please refer to these talks:

  * [Adrin Jalali: Let’s exploit pickle, and skops to the rescue! | PyData Amsterdam 2023](<https://www.youtube.com/watch?v=9w_H5OSTO9A>).

  * [Alex Gaynor: Pickles are for Delis, not Software - PyCon 2014](<https://pyvideo.org/video/2566/pickles-are-for-delis-not-software>).




### 10.5.1. Replicating the training environment in production

If the versions of the dependencies used may differ from training to production, it may result in unexpected behaviour and errors while using the trained model. To prevent such situations it is recommended to use the same dependencies and versions in both the training and production environment. These transitive dependencies can be pinned with the help of package management tools like `pip`, `mamba`, `conda`, `poetry`, `conda-lock`, `pixi`, etc.

It is not always possible to load a model trained with older versions of the scikit-learn library and its dependencies in an updated software environment. Instead, you might need to retrain the model with the new versions of all the libraries. So when training a model, it is important to record the training recipe (e.g. a Python script) and training set information, and metadata about all the dependencies to be able to automatically reconstruct the same training environment for the updated software.

InconsistentVersionWarning

When an estimator is loaded with a scikit-learn version that is inconsistent with the version the estimator was pickled with, an [`InconsistentVersionWarning`](<modules/generated/sklearn.exceptions.InconsistentVersionWarning.html#sklearn.exceptions.InconsistentVersionWarning> "sklearn.exceptions.InconsistentVersionWarning") is raised. This warning can be caught to obtain the original version the estimator was pickled with:
    
    
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.simplefilter("error", InconsistentVersionWarning)
    
    try:
        with open("model_from_previous_version.pickle", "rb") as f:
            est = pickle.load(f)
    except InconsistentVersionWarning as w:
        print(w.original_sklearn_version)
    

### 10.5.2. Serving the model artifact

The last step after training a scikit-learn model is serving the model. Once the trained model is successfully loaded, it can be served to manage different prediction requests. This can involve deploying the model as a web service using containerization, or other model deployment strategies, according to the specifications.

## 10.6. Summarizing the key points

Based on the different approaches for model persistence, the key points for each approach can be summarized as follows:

  * `ONNX`: It provides a uniform format for persisting any machine learning or deep learning model (other than scikit-learn) and is useful for model inference (predictions). It can however, result in compatibility issues with different frameworks.

  * [`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)"): Trained scikit-learn models can be easily shared and put into production using [`skops.io`](<https://skops.readthedocs.io/en/stable/modules/classes.html#module-skops.io> "\(in skops\)"). It is more secure compared to alternate approaches based on [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") because it does not load arbitrary code unless explicitly asked for by the user. Such code needs to be packaged and importable in the target Python environment.

  * [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)"): Efficient memory mapping techniques make it faster when using the same persisted model in multiple Python processes when using `mmap_mode="r"`. It also gives easy shortcuts to compress and decompress the persisted object without the need for extra code. However, it may trigger the execution of malicious code when loading a model from an untrusted source as any other pickle-based persistence mechanism.

  * [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"): It is native to Python and most Python objects can be serialized and deserialized using [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)"), including custom Python classes and functions as long as they are defined in a package that can be imported in the target environment. While [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") can be used to easily save and load scikit-learn models, it may trigger the execution of malicious code while loading a model from an untrusted source. [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") can also be very efficient memorywise if the model was persisted with `protocol=5` but it does not support memory mapping.

  * [cloudpickle](<https://github.com/cloudpipe/cloudpickle>): It has comparable loading efficiency as [`pickle`](<https://docs.python.org/3/library/pickle.html#module-pickle> "\(in Python v3.14\)") and [`joblib`](<https://joblib.readthedocs.io/en/latest/index.html#module-joblib> "\(in joblib v1.6.dev0\)") (without memory mapping), but offers additional flexibility to serialize custom Python code such as lambda expressions and interactively defined functions and classes. It might be a last resort to persist pipelines with custom Python components such as a [`sklearn.preprocessing.FunctionTransformer`](<modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer> "sklearn.preprocessing.FunctionTransformer") that wraps a function defined in the training script itself or more generally outside of any importable Python package. Note that [cloudpickle](<https://github.com/cloudpipe/cloudpickle>) offers no forward compatibility guarantees and you might need the same version of [cloudpickle](<https://github.com/cloudpipe/cloudpickle>) to load the persisted model along with the same version of all the libraries used to define the model. As the other pickle-based persistence mechanisms, it may trigger the execution of malicious code while loading a model from an untrusted source.



