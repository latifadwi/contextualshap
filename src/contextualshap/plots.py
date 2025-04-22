import copy
import shap
import io
import matplotlib.pyplot as plt
import base64
import json
from openai import OpenAI
from .common import _table, languages


def _explain_waterfall(image, explanation: shap.Explanation, feature_aliases=None, feature_descriptions=None, additional_background=None, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en'):
    if language not in languages:
        raise ValueError("Language must be one of: " + ", ".join(languages))

    if hasattr(explanation.base_values, "__len__"):
        # TODO: document what should happen if base_values is a vector
        raise ValueError("Explanation base values is a list, currently unsupported")
    else:
        prediction = explanation.base_values

    client = OpenAI(api_key=openai_api_key)

    bi = base64.b64encode(image).decode()
    prompt_features = []

    i = 0
    for f in explanation.feature_names:
        desc = ''
        if feature_descriptions is not None and f in feature_descriptions:
            desc = feature_descriptions[f]

        alias = ''
        if feature_aliases is not None and f in feature_aliases:
            alias = feature_aliases[f]

        prompt_features.append(
            {'Feature Name': f, 'Feature Alias': alias, 'Feature Description': desc, 'SHAP Value': str(explanation.values[i]), 'Sample Value': str(explanation.data[i])})
        i = i + 1

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
            The given image is a waterfall plot of a single prediction sample in the dataset. The result of the prediction of this sample of the dataset is {prediction}.
            {'' if len(prompt_features) == 0 else f'Alias and description of the feature names:\n{_table(prompt_features)}\nThe feature description sometimes explain what the values mean, and you must include the explanation for the values in the result.\n'}
            {'' if additional_background is None else f'Context background of this model to be included in the explanation: {additional_background}.'}
            Reply in {languages[language]} language. Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know.
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

def waterfall(explanation: shap.Explanation, feature_aliases=None, feature_descriptions=None, additional_background=None, show=True, explain=True, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en', **kwargs):
    """
    Displays a SHAP waterfall plot. This is a utility wrapper function that accepts feature aliases dictionary
    to easily alias some feature names that are otherwise retrieved by default through explanation.feature_names.
    By setting explain to True, this function will also return a string of narration containing explanation about the waterfall plot.
    **kwargs is passed to shap.plots.waterfall function to modify the function.

    :param explanation: a shap.Explanation instance retrieved from calling shap explainer.
    :param feature_aliases: an optional dictionary mapping of old feature name to new feature name.
    :param feature_descriptions: an optional dictionary mapping of old feature name to description.
    :param additional_background: an optional background string to be given to GPT to enhance explanation.
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

        return _explain_waterfall(data, explanation, feature_aliases, feature_descriptions, additional_background, openai_api_key, gpt_model, language)
    else:
        if show:
            plt.show()

def _explain_bar(image, feature_names, feature_aliases=None, feature_descriptions=None, additional_background=None, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en'):
    if language not in languages:
        raise ValueError("Language must be one of: " + ", ".join(languages))

    client = OpenAI(api_key=openai_api_key)

    bi = base64.b64encode(image).decode()
    prompt_features = []

    for f in feature_names:
        desc = ''
        if feature_descriptions is not None and f in feature_descriptions:
            desc = feature_descriptions[f]

        alias = ''
        if feature_aliases is not None and f in feature_aliases:
            alias = feature_aliases[f]

        prompt_features.append(
            {'Feature Name': f, 'Feature Alias': alias, 'Feature Description': desc})

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
            Your job is to output an easy explanation about the image in the context of the SHAP values to better explain to readers the meaning of the bar plot.
            The given image is a bar plot of a features in the dataset and their corresponding average SHAP values.
            {'' if len(prompt_features) == 0 else f'Alias and description of the feature names:\n{_table(prompt_features)}\n'}
            {'' if additional_background is None else f'Context background of this model to be included in the explanation: {additional_background}.'}
            Reply in {languages[language]} language. Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know.
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

def bar(shap_values, feature_aliases=None, feature_descriptions=None, additional_background=None, explain=True, show=True, openai_api_key=None, gpt_model = 'gpt-4o', language = 'en', **kwargs):
    """
        Displays a SHAP bar plot. This is a utility wrapper function that accepts feature aliases dictionary
        to easily alias some feature names that are otherwise retrieved by default through shap_values.feature_names.
        By setting explain to True, this function will also return a string of narration containing explanation about the bar plot.
        **kwargs is passed to shap.plots.bar function to modify the function.

        :param shap_values: a shap.Explanation or shap.Cohorts or dictionary of shap.Explanation instance retrieved from calling shap explainer.
        :param feature_aliases: an optional dictionary mapping of old feature name to new feature name.
        :param feature_descriptions: an optional dictionary mapping of old feature name to description.
        :param additional_background: an optional background string to be given to GPT to enhance explanation.
        :param show: setting this to false will not call plot.show to show the plot.
        :param explain: setting this to false will not call OpenAI API to retrieve the plot explanation, so API key, GPT model, and language parameters are not used. Nothing will be returned if explain is False.
        :param openai_api_key: an OpenAI API key to use for API calls.
        :param gpt_model: a GPT model to use.
        :param language: a language code to use.
        :return: anything returned  by shap.plots.waterfall, especially in the case of setting `show=False`.
        """

    if feature_aliases is None:
        feature_aliases = {}

    original_feature_names = []
    feature_names = []
    if isinstance(shap_values, shap.Explanation):
        for f in shap_values.feature_names:
            if f in feature_aliases:
                feature_names.append(feature_aliases[f])
            else:
                feature_names.append(f)
        nsv = copy.deepcopy(shap_values)
        nsv.feature_names = feature_names
        original_feature_names = shap_values.feature_names
    elif isinstance(shap_values, shap.Cohorts):
        cohorts = copy.deepcopy(shap_values.cohorts)
        for label, exp in cohorts.items():
            original_feature_names = exp.feature_names
            for f in exp.feature_names:
                if f in feature_aliases:
                    feature_names.append(feature_aliases[f])
                else:
                    feature_names.append(f)
            exp.feature_names = feature_names
        nsv = cohorts
    elif isinstance(shap_values, dict):
        cohorts = copy.deepcopy(shap_values)
        for label, exp in cohorts.items():
            original_feature_names = exp.feature_names
            for f in exp.feature_names:
                if f in feature_aliases:
                    feature_names.append(feature_aliases[f])
                else:
                    feature_names.append(f)
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

        return _explain_bar(data, original_feature_names, feature_aliases, feature_descriptions, additional_background, openai_api_key, gpt_model, language)
    else:
        if show:
            plt.show()