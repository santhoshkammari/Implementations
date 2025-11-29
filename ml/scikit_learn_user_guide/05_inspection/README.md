# 5\. Inspection

Predictive performance is often the main goal of developing machine learning models. Yet summarizing performance with an evaluation metric is often insufficient: it assumes that the evaluation metric and test dataset perfectly reflect the target domain, which is rarely true. In certain domains, a model needs a certain level of interpretability before it can be deployed. A model that is exhibiting performance issues needs to be debugged for one to understand the model’s underlying issue. The [`sklearn.inspection`](<api/sklearn.inspection.html#module-sklearn.inspection> "sklearn.inspection") module provides tools to help understand the predictions from a model and what affects them. This can be used to evaluate assumptions and biases of a model, design a better model, or to diagnose issues with model performance.

Examples

  * [Common pitfalls in the interpretation of coefficients of linear models](<auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#sphx-glr-auto-examples-inspection-plot-linear-model-coefficient-interpretation-py>)




  * [5.1. Partial Dependence and Individual Conditional Expectation plots](<modules/partial_dependence.html>)
    * [5.1.1. Partial dependence plots](<modules/partial_dependence.html#partial-dependence-plots>)
    * [5.1.2. Individual conditional expectation (ICE) plot](<modules/partial_dependence.html#individual-conditional-expectation-ice-plot>)
    * [5.1.3. Mathematical Definition](<modules/partial_dependence.html#mathematical-definition>)
    * [5.1.4. Computation methods](<modules/partial_dependence.html#computation-methods>)
  * [5.2. Permutation feature importance](<modules/permutation_importance.html>)
    * [5.2.1. Outline of the permutation importance algorithm](<modules/permutation_importance.html#outline-of-the-permutation-importance-algorithm>)
    * [5.2.2. Relation to impurity-based importance in trees](<modules/permutation_importance.html#relation-to-impurity-based-importance-in-trees>)
    * [5.2.3. Misleading values on strongly correlated features](<modules/permutation_importance.html#misleading-values-on-strongly-correlated-features>)


