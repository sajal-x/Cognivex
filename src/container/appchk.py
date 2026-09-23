from fetch_data import fetch_data_by_id,fetch_all_id
from analyzer_model import analyze
import shap
import matplotlib.pyplot as plt
import numpy as np

class_idx=0
all_id=fetch_all_id()
selected_id=all_id[20]["id"]
fetched_data=fetch_data_by_id(selected_id)
prediction=analyze(fetched_data)
subtype=prediction["subtype"]
allclassprobabilities=prediction["probabilities"]

shapdata=prediction["shapdata"]
shap_values_array = shapdata["values"]
base_values = shapdata["basevalue"]
data_values = shapdata["datavalues"]
features = shapdata["features"]

data_matrix = np.array(data_values).reshape(1, -1)
shap_values_array = np.array(shapdata["values"])


print(f'{selected_id}\n{fetched_data}')
print(subtype)
print(allclassprobabilities)


explanation_obj = shap.Explanation(
    values=shap_values_array[-1, :, class_idx],  
    base_values=base_values[class_idx],
    data=data_values,
    feature_names=features,
)


print("Generating SHAP Bar Plot...")
shap.plots.bar(explanation_obj, max_display=10)


print("Generating SHAP Force Plot...")

shap.plots.force(
    base_values[class_idx],
    shap_values_array[-1, :, class_idx],  
    data_values,
    feature_names=features,
    matplotlib=True,
)



print("Generating SHAP Decision Plot...")
plt.figure()
shap.decision_plot(
    base_values[class_idx],
    shap_values_array[:, :, class_idx],
    features=features,
)
plt.show()

