from openai import OpenAI
import shap
import json
import pandas as pd

def _table(list_of_dicts):
    """Converts a list of dictionaries to markdown-formatted table.

    list_of_dicts -- Each dict is a row
    """
    v = ""
    # Make a string of all the keys in the first dict with pipes before after and between each key
    head = f"| {" | ".join(map(str, list_of_dicts[0].keys()))} |"
    # Make a header separator line with dashes instead of key names
    sep = f"{"|-----" * len(list_of_dicts[0].keys())}|"
    # Add the header row and separator to the table
    v += head + "\n"
    v += sep + "\n"

    for row in list_of_dicts:
        md_row = ""
        for key, col in row.items():
            md_row += f"| {str(col)} "
        v += md_row + "|\n"
    return v


def explain(shap_values: list[shap.Explanation], feature_aliases: dict, feature_descriptions: dict, openai_api_key = None, gpt_model = 'gpt-4o', additional_background = None):
    prompt_feature_aliases = []
    prompt_shap_values = []
    for f in shap_values[0].feature_names:
        desc = ''
        if f in feature_descriptions:
            desc = feature_descriptions[f]

        if f in feature_aliases:
            prompt_feature_aliases.append(
                {'Feature Name': f, 'Feature Alias': feature_aliases[f], 'Feature Description': desc})
        else:
            prompt_feature_aliases.append({'Feature Name': f, 'Feature Alias': f, 'Feature Description': desc})

    for x in range(len(shap_values)):
        for i in range(len(shap_values[x].data)):
            prompt_shap_values.append(
                {'Sample Number': str(x), 'Feature Name': shap_values[x].feature_names[i], 'Input Value': shap_values[x].data[i],
                 'SHAP Value': shap_values[x].values[i]})

    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": f"""
    SHAP refers to SHapley Additive exPlanations. Refer to the "A Unified Approach to Interpreting Model Predictions" paper by Scott Lundberg. This is about AI model training.
    Your job is to output an explanation about each feature according to the SHAP values to better explain to readers the meaning of these SHAP values for each of the features and the result of the AI model.
    This is a table of feature names of the dataset, their aliases, and the description of the feature. If there is no description or alias, interpret the feature name yourself.
    {_table(prompt_feature_aliases)}
    You are now given a few samples of the AI model prediction, consists of the input value and SHAP value for each feature. 
    {_table(prompt_shap_values)}
    {'' if additional_background is None else f'Context background of this model to be included in the explanation: {additional_background}.'}
    Also add a summary of everything that is given.
    Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know. Output is only a JSON object with a string field `summary` and `features`, which is an array of JSON with field name 'feature_name' for the feature name, 'description' for the description that you interpreted, and 'explanation' for the explanation. Do not enclose the JSON in markdown code."""
            }
        ]
    )

    response = json.loads(completion.choices[0].message.content)
    return response['summary'], pd.DataFrame(response['features'])