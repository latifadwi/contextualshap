# contextualshap

The `contextualshap` package provides additional functionalities on top of the widely used `shap` package.
The main function of SHAP is to provide explanations for AI models (explainable AI/XAI). The SHAP package itself
has provided many functionalities to calculate and visualize SHAP values, and this package provides functionalities
to add additional narration in addition of SHAP visualizations, that are generated with GPT model to better improve explanation power provided
by SHAP.

## Usage

Following the basic example from SHAP, we can create a basic Linear Regression model to demonstrate the
contextualshap ability to generate a new narration based on the calculated SHAP values.

```python
import shap
import sklearn

# a classic housing price dataset
X, y = shap.datasets.california(n_points=1000)

X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

# a simple linear model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# compute the SHAP values for the linear model
explainer = shap.Explainer(model.predict, X100)
shap_values = explainer(X)
```

From the generated `shap_values`, per-feature narration can be generated.

```python
import contextualshap.gpt
import contextualshap.plots

# Optional feature aliases
feature_aliases = {
    'MedInc': 'Median Income'
}

# Optional feature descriptions to GPT ability
feature_descriptions = {
    'MedInc': 'Median household income',
    'AveRooms': "Average number of rooms",
    'AveBedrms': "Average number of bedrooms",
    'AveOccup': 'Average number of occupancies',
}

# Draw a waterfall plot with feature aliases, taking the first SHAP value for example
contextualshap.plots.waterfall(shap_values[0], feature_aliases=feature_aliases, max_display=14)

# Explain using the first 10 sample SHAP values
summary, feature_explanations = contextualshap.gpt.explain(shap_values[:10], feature_aliases, feature_descriptions, openai_api_key='<your-api-key>', additional_background='This model trains on the California housing price dataset. It tries to predict the house price from the features.', language='id')
print(summary)
print(feature_explanations)
```

### Explaining a SHAP Waterfall Plot

The `plots.waterfall` function is a wrapper function around `shap.plots.waterfall` which supports `feature_aliases` parameter
to change feature names and `explain` parameter configures the function to also retrieve GPT model explanation about the plot.
When `explain` is set to True (the default), it will generate an explanation of the plot which is returned by the function.

```python
import contextualshap.plots

# Optional feature aliases
feature_aliases = {
    'MedInc': 'Median Income'
}

# Draw a waterfall plot with feature aliases, taking the first SHAP value for example
explanation = contextualshap.plots.waterfall(shap_values[0], feature_aliases=feature_aliases, max_display=14, openai_api_key='<your-api-key>')
print(explanation)
```

Keep in mind that to plot a waterfall plot, a single SHAP value (a single instance of prediction) is used. This means the
explanation generated will also explain only a single prediction. To better enhance the explanation capability, the bar
plot can also be used.

### Explaining Feature Importance Bar Plot

The `plots.bar` function wraps around `shap.plots.bar` function. Similar to waterfall, it takes `explain` parameter
which by default is set to True. This function will return a string of explanation when set to explain the bar plot.
The bar plot can take a list of SHAP values, in which it will retrieve and average out
all the SHAP values for each feature. The explanation provided by this function explains how each feature is more
important for prediction than the others.

```python
import contextualshap.plots

# Optional feature aliases
feature_aliases = {
    'MedInc': 'Median Income'
}

# Draw a bar plot with feature aliases, taking the first SHAP value for example
explanation = contextualshap.plots.bar(shap_values, feature_aliases=feature_aliases, max_display=14, openai_api_key='<your-api-key>')
print(explanation)
```

## Limitation and TODO

The currently supported waterfall/bar plots apply only for single output model explainers. That is, the model should output
a scalar value instead of a vector of values. To explain more complex model, see also `shap.DeepExplainer` from the SHAP
package.