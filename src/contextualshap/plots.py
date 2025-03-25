import copy
import shap

def waterfall(explanation: shap.Explanation, feature_aliases=None, **kwargs):
    """
    Displays a SHAP waterfall plot. This is a utility wrapper function that accepts feature aliases dictionary
    to easily alias some feature names that are otherwise retrieved by default through explanation.feature_names.

    :param explanation: a shap.Explanation instance retrieved from calling shap explainer.
    :param feature_aliases: a dictionary mapping of old feature name to new feature name.
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

    return shap.plots.waterfall(nsv, **kwargs)