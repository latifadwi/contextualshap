import copy
import shap
import io
import matplotlib.pyplot as plt
import base64
import json
from openai import OpenAI

def _explain_waterfall(image, base_value, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en'):
    client = OpenAI(api_key=openai_api_key)

    bi = base64.b64encode(image).decode()

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
            SHAP refers to SHapley Additive exPlanations. Refer to the "A Unified Approach to Interpreting Model Predictions" paper by Scott Lundberg. This is about AI model training.
            Your job is to output an easy explanation about the image in the context of the SHAP values to better explain to readers the meaning of the waterfall plot.
            The given image is a waterfall plot of a single prediction sample in the dataset. The result of the prediction of this sample of the dataset is {base_value}.
            Reply in {language} language. Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know.
            Output is only a JSON object with a string field `explanation` containing the explanation.
            Do not enclose the JSON in markdown code."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{bi}"
                        }
                    }
                ]
            }
        ]
    )

    response = json.loads(completion.choices[0].message.content)
    return response['explanation']

def waterfall(explanation: shap.Explanation, feature_aliases=None, show=True, explain=True, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en', **kwargs):
    """
    Displays a SHAP waterfall plot. This is a utility wrapper function that accepts feature aliases dictionary
    to easily alias some feature names that are otherwise retrieved by default through explanation.feature_names.
    By setting explain to True, this function will also return a string of narration containing explanation about the waterfall plot.
    **kwargs is passed to shap.plots.waterfall function to modify the function.

    :param explanation: a shap.Explanation instance retrieved from calling shap explainer.
    :param feature_aliases: a dictionary mapping of old feature name to new feature name.
    :param show: setting this to false will not call plot.show to show the plot.
    :param explain: setting this to false will not call OpenAI API to retrieve the plot explanation, so API key, GPT model, and language parameters are not used. Nothing will be returned if explain is False.
    :param openai_api_key: an OpenAI API key to use for API calls.
    :param gpt_model: a GPT model to use.
    :param language: a language code to use.
    :return: anything returned  by shap.plots.waterfall, especially in the case of setting `show=False`.
    """
    if feature_aliases is None:
        feature_aliases = {}

    feature_names = []
    for f in explanation.feature_names:
        if f in feature_aliases:
            feature_names.append(feature_aliases[f])
        else:
            feature_names.append(f)
    nsv = copy.deepcopy(explanation)
    nsv.feature_names = feature_names

    shap.plots.waterfall(nsv, show=False, **kwargs)

    if explain:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = buf.getvalue()
        buf.close()

        if show:
            plt.show()

        if hasattr(nsv.base_values, "__len__"):
            # TODO: document what should happen if base_values is a vector
            pass
        else:
            prediction = nsv.base_values

        return _explain_waterfall(data, prediction, openai_api_key, gpt_model, language)
    else:
        if show:
            plt.show()

def _explain_bar(image, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en'):
    client = OpenAI(api_key=openai_api_key)

    bi = base64.b64encode(image).decode()

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
            SHAP refers to SHapley Additive exPlanations. Refer to the "A Unified Approach to Interpreting Model Predictions" paper by Scott Lundberg. This is about AI model training.
            Your job is to output an easy explanation about the image in the context of the SHAP values to better explain to readers the meaning of the waterfall plot.
            The given image is a bar plot of a features in the dataset and their corresponding average SHAP values.
            Reply in {language} language. Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know.
            Output is only a JSON object with a string field `explanation` containing the explanation.
            Do not enclose the JSON in markdown code."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{bi}"
                        }
                    }
                ]
            }
        ]
    )

    response = json.loads(completion.choices[0].message.content)
    return response['explanation']

def bar(shap_values, feature_aliases=None, explain=True, show=True, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en', **kwargs):
    if feature_aliases is None:
        feature_aliases = {}

    feature_names = []
    if isinstance(shap_values, shap.Explanation):
        for f in shap_values.feature_names:
            if f in feature_aliases:
                feature_names.append(feature_aliases[f])
            else:
                feature_names.append(f)
        nsv = copy.deepcopy(shap_values)
        nsv.feature_names = feature_names
    elif isinstance(shap_values, shap.Cohorts):
        cohorts = copy.deepcopy(shap_values.cohorts)
        for label, exp in cohorts.items():
            exp.feature_names = feature_names
        nsv = cohorts
    elif isinstance(shap_values, dict):
        cohorts = copy.deepcopy(shap_values)
        for label, exp in cohorts.items():
            exp.feature_names = feature_names
        nsv = cohorts
    else:
        emsg = (
            "The shap_values argument must be an Explanation object, Cohorts "
            "object, or dictionary of Explanation objects!"
        )
        raise TypeError(emsg)

    shap.plots.bar(nsv, show=False, **kwargs)

    if explain:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        data = buf.getvalue()
        buf.close()

        if show:
            plt.show()

        return _explain_bar(data, openai_api_key, gpt_model, language)
    else:
        if show:
            plt.show()