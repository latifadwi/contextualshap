import unittest
import shap
import sklearn
from src.contextualshap import plots

class PlotsTestCase(unittest.TestCase):
    def test_waterfall(self):
        # a classic housing price dataset
        x, y = shap.datasets.california(n_points=1000)

        x100 = shap.utils.sample(x, 100)  # 100 instances for use as the background distribution

        # a simple linear model
        model = sklearn.linear_model.LinearRegression()
        model.fit(x, y)

        # compute the SHAP values for the linear model
        explainer = shap.Explainer(model.predict, x100)
        shap_values = explainer(x)

        sample_ind = 0
        # This will fail without a valid OpenAI API key
        # print(plots.waterfall(shap_values[sample_ind], max_display=14, openai_api_key=''))

    def test_bar(self):
        # a classic housing price dataset
        x, y = shap.datasets.california(n_points=1000)

        x100 = shap.utils.sample(x, 100)  # 100 instances for use as the background distribution

        # a simple linear model
        model = sklearn.linear_model.LinearRegression()
        model.fit(x, y)

        # compute the SHAP values for the linear model
        explainer = shap.Explainer(model.predict, x100)
        shap_values = explainer(x)

        feature_aliases = {
            'MedInc': 'Median Income'
        }

        # This will fail without a valid OpenAI API key
        # print(plots.bar(shap_values, max_display=14, feature_aliases=feature_aliases, openai_api_key=''))