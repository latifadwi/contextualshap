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