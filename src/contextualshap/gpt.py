from openai import OpenAI
import shap
import json
import pandas as pd

languages = {
    'aa': 'Afar',
    'ab': 'Abkhazian',
    'af': 'Afrikaans',
    'ak': 'Akan',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'an': 'Aragonese',
    'hy': 'Armenian',
    'as': 'Assamese',
    'av': 'Avaric',
    'ae': 'Avestan',
    'ay': 'Aymara',
    'az': 'Azerbaijani',
    'ba': 'Bashkir',
    'bm': 'Bambara',
    'be': 'Belarusian',
    'bn': 'Bengali',
    'bh': 'Bihari languages',
    'bi': 'Bislama',
    'bo': 'Tibetan',
    'bs': 'Bosnian',
    'br': 'Breton',
    'bg': 'Bulgarian',
    'my': 'Burmese',
    'ca': 'Catalan; Valencian',
    'ch': 'Chamorro',
    'ce': 'Chechen',
    'zh': 'Chinese',
    'cu': 'Church Slavic; Old Slavonic; Church Slavonic; Old Bulgarian; Old Church Slavonic',
    'cv': 'Chuvash',
    'kw': 'Cornish',
    'co': 'Corsican',
    'cr': 'Cree',
    'cy': 'Welsh',
    'cs': 'Czech',
    'da': 'Danish',
    'dv': 'Divehi; Dhivehi; Maldivian',
    'nl': 'Dutch; Flemish',
    'dz': 'Dzongkha',
    'el': 'Greek: Modern 1453-',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'eu': 'Basque',
    'ee': 'Ewe',
    'fo': 'Faroese',
    'fa': 'Persian',
    'fj': 'Fijian',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ff': 'Fulah',
    'Ga': 'Georgian',
    'de': 'German',
    'gd': 'Gaelic; Scottish Gaelic',
    'ga': 'Irish',
    'gl': 'Galician',
    'gv': 'Manx',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'ht': 'Haitian; Haitian Creole',
    'ha': 'Hausa',
    'he': 'Hebrew',
    'hz': 'Herero',
    'hi': 'Hindi',
    'ho': 'Hiri Motu',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'ig': 'Igbo',
    'is': 'Icelandic',
    'io': 'Ido',
    'ii': 'Sichuan Yi; Nuosu',
    'iu': 'Inuktitut',
    'ie': 'Interlingue; Occidental',
    'ia': 'Interlingua International Auxiliary Language Association',
    'id': 'Indonesian',
    'ik': 'Inupiaq',
    'it': 'Italian',
    'jv': 'Javanese',
    'ja': 'Japanese',
    'kl': 'Kalaallisut; Greenlandic',
    'kn': 'Kannada',
    'ks': 'Kashmiri',
    'ka': 'Georgian',
    'kr': 'Kanuri',
    'kk': 'Kazakh',
    'km': 'Central Khmer',
    'ki': 'Kikuyu; Gikuyu',
    'rw': 'Kinyarwanda',
    'ky': 'Kirghiz; Kyrgyz',
    'kv': 'Komi',
    'kg': 'Kongo',
    'ko': 'Korean',
    'kj': 'Kuanyama; Kwanyama',
    'ku': 'Kurdish',
    'lo': 'Lao',
    'la': 'Latin',
    'lv': 'Latvian',
    'li': 'Limburgan; Limburger; Limburgish',
    'ln': 'Lingala',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish; Letzeburgesch',
    'lu': 'Luba-Katanga',
    'lg': 'Ganda',
    'mh': 'Marshallese',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'Mi': 'Micmac',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'mt': 'Maltese',
    'mn': 'Mongolian',
    'mi': 'Maori',
    'ms': 'Malay',
    'na': 'Nauru',
    'nv': 'Navajo; Navaho',
    'nr': 'Ndebele: South; South Ndebele',
    'nd': 'Ndebele: North; North Ndebele',
    'ng': 'Ndonga',
    'ne': 'Nepali',
    'nn': 'Norwegian Nynorsk; Nynorsk: Norwegian',
    'nb': 'Bokmål: Norwegian; Norwegian Bokmål',
    'no': 'Norwegian',
    'oc': 'Occitan post 1500',
    'oj': 'Ojibwa',
    'or': 'Oriya',
    'om': 'Oromo',
    'os': 'Ossetian; Ossetic',
    'pa': 'Panjabi; Punjabi',
    'pi': 'Pali',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ps': 'Pushto; Pashto',
    'qu': 'Quechua',
    'rm': 'Romansh',
    'ro': 'Romanian; Moldavian; Moldovan',
    'rn': 'Rundi',
    'ru': 'Russian',
    'sg': 'Sango',
    'sa': 'Sanskrit',
    'si': 'Sinhala; Sinhalese',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'se': 'Northern Sami',
    'sm': 'Samoan',
    'sn': 'Shona',
    'sd': 'Sindhi',
    'so': 'Somali',
    'st': 'Sotho: Southern',
    'es': 'Spanish; Castilian',
    'sc': 'Sardinian',
    'sr': 'Serbian',
    'ss': 'Swati',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ty': 'Tahitian',
    'ta': 'Tamil',
    'tt': 'Tatar',
    'te': 'Telugu',
    'tg': 'Tajik',
    'tl': 'Tagalog',
    'th': 'Thai',
    'ti': 'Tigrinya',
    'to': 'Tonga Tonga Islands',
    'tn': 'Tswana',
    'ts': 'Tsonga',
    'tk': 'Turkmen',
    'tr': 'Turkish',
    'tw': 'Twi',
    'ug': 'Uighur; Uyghur',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'uz': 'Uzbek',
    've': 'Venda',
    'vi': 'Vietnamese',
    'vo': 'Volapük',
    'wa': 'Walloon',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'yo': 'Yoruba',
    'za': 'Zhuang; Chuang',
    'zu': 'Zulu'
}

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


def explain(shap_values: list[shap.Explanation], feature_aliases: dict, feature_descriptions: dict, openai_api_key = None, gpt_model = 'gpt-4o', additional_background = None, language = 'en'):
    """
    Generates an explanation for each features according to the SHAP values. The generated narration can be displayed to
    end-users to better help end-users understand about the result of the SHAP values. Can be paired with SHAP visualizations
    to increase user experience.

    :param shap_values: a list of SHAP values, please take only a few SHAP values to avoid OpenAI API token limit
    :param feature_aliases: an optional dictionary containing alias per feature, to increase explanation clarity
    :param feature_descriptions: an optional dictionary containing description per feature, to increase explanation clarity
    :param openai_api_key: OpenAI API key string
    :param gpt_model: the OpenAI GPT model
    :param additional_background: additional narration containing background story of the model to increase explanation power
    :param language: the language of the response
    :return: summary (a string) anf a list of dictionary containing descriptions for each feature names
    """

    if language not in languages:
        raise ValueError("Language must be one of: " + ", ".join(languages))

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
    Reply in {language} language. Give explanation for each feature name and the SHAP values for amateur readers. Also add some more explanation or context that you know. Output is only a JSON object with a string field `summary` and `features`, which is an array of JSON with field name 'feature_name' for the feature name, 'description' for the description that you interpreted, and 'explanation' for the explanation. Do not enclose the JSON in markdown code."""
            }
        ]
    )

    response = json.loads(completion.choices[0].message.content)
    return response['summary'], pd.DataFrame(response['features'])