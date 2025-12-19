import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import set_config

set_config(display='diagram')

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

cols_to_clean = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', MinMaxScaler()), 
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski']
}

print("Starting Grid Search... (This might take a moment)")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=10,
    scoring='accuracy', 
    n_jobs=-1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)

print("\n=== PIPELINE VISUALIZATION ===")
print(pipeline) 

results_df = pd.DataFrame(grid_search.cv_results_)

heatmap_data = results_df[[
    'param_knn__n_neighbors', 
    'param_knn__metric', 
    'param_knn__weights', 
    'mean_test_score'
]]

print("\n=== TOP 5 PERFORMING HYPERPARAMETERS ===")
print(heatmap_data.sort_values(by='mean_test_score', ascending=False).head(5))

plt.figure(figsize=(14, 6))

g = sns.FacetGrid(heatmap_data, col="param_knn__weights", height=5, aspect=1.2)
g.map_dataframe(
    sns.lineplot, 
    x="param_knn__n_neighbors", 
    y="mean_test_score", 
    hue="param_knn__metric", 
    style="param_knn__metric",
    markers=True, 
    dashes=False,
    linewidth=2
)
g.add_legend(title="Metric")
g.set_axis_labels("Number of Neighbors (K)", "Cross-Validation Accuracy")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Impact of K, Metric, and Weight on Model Accuracy', fontsize=16)
plt.show()

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"\nOptimal Hyperparameters: {grid_search.best_params_}")
print(f"Final Test Accuracy (Unseen Data): {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Healthy', 'Diabetic'],
            yticklabels=['Healthy', 'Diabetic'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix\n(Best Model: {grid_search.best_params_["knn__metric"]} distance, K={grid_search.best_params_["knn__n_neighbors"]})', fontsize=14)
plt.show()
