import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.data_io import save_dataframe
from src.preprocessing.segmentation import segment_lesion
from src.preprocessing.color_features import extract_color_statistics
from src.preprocessing.glcm_features import extract_glcm_features
from src.transformation.adjust_features import adjusted_dataframe
from src.modeling.train_models import get_models
from src.modeling.evaluate_models import evaluate_model
from src.modeling.cross_validation import svm_cross_validation

DATASET_PATH = "./data/images"
LABELS_PATH = "./data/labels.csv"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels_df = pd.read_csv(LABELS_PATH)
features_list = []
labels_list = []

for idx, row in labels_df.iterrows():
    image_name = row["image"]
    label = row["label"]
    img_path = os.path.join(DATASET_PATH, image_name)
    image = cv2.imread(img_path)
    if image is None:
        continue
    mask, segmented = segment_lesion(image)
    color_stats = extract_color_statistics(segmented)
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    glcm_stats = extract_glcm_features(gray)
    features = {**color_stats, **glcm_stats}
    features_list.append(features)
    labels_list.append(label)

df = pd.DataFrame(features_list)
df_labels = pd.DataFrame(labels_list, columns=["label"])
save_dataframe(df, os.path.join(OUTPUT_DIR, "features_raw.xlsx"))
save_dataframe(df_labels, os.path.join(OUTPUT_DIR, "labels.xlsx"))
df_adjusted = adjusted_dataframe(df)
save_dataframe(df_adjusted, os.path.join(OUTPUT_DIR, "features_adjusted.xlsx"))

X_train, X_test, y_train, y_test = train_test_split(df_adjusted, df_labels, test_size=0.3, random_state=42)
models = get_models()
for name, model in models.items():
    model.fit(X_train, y_train.values.ravel())
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Modelo: {name}, Accuracy: {metrics['accuracy']}, AUC: {metrics['auc']}")

best_params, best_score = svm_cross_validation(
    X_train, y_train.values.ravel(),
    c_values=[1, 2, 3], gamma_values=[0.01, 0.1, 1],
    kernels=['linear', 'rbf'], cv=5
)
print(f"Mejores parámetros SVM: {best_params}, Accuracy CV: {best_score}")
