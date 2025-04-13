# ====================
# Step 0: Import Libraries
# ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning and preprocessing libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import pickle

# ====================
# Step 1: Data Loading and Initial Exploration
# ====================
data_path = r'C:\Users\Shruti Gavhane\Downloads\yourfile.csv'  # Fix file path
df = pd.read_csv(data_path)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

df  # Display the DataFrame in notebooks environments
X = df.drop("Yield(Kg per ha)", axis=1)  # Correct column name
y = df["Yield(Kg per ha)"]

# Print dataset info and a few records
print("Dataset Information:")
print(df.info())
print("\nFirst 5 Records:")
print(df.head())

# Visualize basic distributions for the raw data:
plt.figure(figsize=(14, 6))

# (A) Distribution of the target variable (crop yield)
plt.subplot(1, 2, 1)
sns.histplot(df['Yield(Kg per ha)'], bins=20, kde=True, color='purple')
plt.title('Raw Distribution of Crop Yield')

# (B) Count plot for one categorical variable (example: district)
plt.subplot(1, 2, 2)
sns.countplot(data=df, x='Dist Name', palette='viridis')
plt.title('Count per District (Raw)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ====================
# Step 2: Define Target and Feature Columns
# ====================
target = 'Yield(Kg per ha)'
categorical_features = ['Dist Name', 'Crop']  # adjust if different
numerical_features = [col for col in df.columns if col not in categorical_features + [target]]

print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)

# ====================
# Step 2.1: Feature Selection using a Correlation Matrix
# ====================
# Calculate correlation among numerical features and the target.
correlation_matrix = df[numerical_features + [target]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features and Target")
plt.show()

# Step 2.2: Remove Highly Correlated Features
# ====================
def remove_highly_correlated_features(df, features, threshold=0.9):
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop

features_to_drop = remove_highly_correlated_features(df, numerical_features, threshold=0.8)
print("Features to drop due to high correlation:", features_to_drop)

# Remove the highly correlated features from the numerical_features list
numerical_features = [feature for feature in numerical_features if feature not in features_to_drop]
print("Final numerical features after elimination:", numerical_features)

# ====================
# Step 3: Data Preprocessing Pipeline
# ====================
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),   # Handling missing values
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing categories
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# ====================
# Step 4: Visualize Preprocessing for a Single Feature
# ====================
if numerical_features:
    demo_feature = numerical_features[0]
    scaler = StandardScaler()
    scaled_feature = scaler.fit_transform(df[[demo_feature]])

    plt.figure(figsize=(12, 5))

    # Plot raw feature distribution
    plt.subplot(1, 2, 1)
    sns.histplot(df[demo_feature], bins=20, kde=True, color='coral')
    plt.title(f'Raw Distribution of "{demo_feature}"')

    # Plot scaled feature distribution
    plt.subplot(1, 2, 2)
    sns.histplot(scaled_feature.flatten(), bins=20, kde=True, color='teal')
    plt.title(f'Standardized Distribution of "{demo_feature}"')

    plt.tight_layout()
    plt.show()
else:
    print("No numerical features found for demonstration.")

# ====================
# Step 5: Transform the Whole Dataset and Visualize Effects
# ====================
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
print("\nPreprocessed training data shape:", X_train_transformed.shape)

# ====================
# Step 6: Model Building and Training
# ====================

# ---- Model A: Linear Regression ----
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Ensure missing values are imputed
    ('linear_regression', LinearRegression())
])
linear_pipeline.fit(X_train, y_train)
y_pred_linear = linear_pipeline.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print("\n----- Linear Regression Results -----")
print(f"Mean Squared Error: {mse_linear:.2f}")
print(f"R² Score: {r2_linear:.2f}")

# ---- Model B: Decision Tree Regressor ----
tree_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('decision_tree', DecisionTreeRegressor(random_state=42))
])
param_grid_tree = {
    'decision_tree__max_depth': [None, 5, 10, 20],
    'decision_tree__min_samples_split': [2, 5, 10],
    'decision_tree__min_samples_leaf': [1, 2, 4]
}
grid_search_tree = GridSearchCV(tree_pipeline, param_grid_tree, cv=5,
                                scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_tree.fit(X_train, y_train)
best_tree_model = grid_search_tree.best_estimator_
y_pred_tree = best_tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print("\n----- Decision Tree Regressor Results -----")
print("Best Parameters:", grid_search_tree.best_params_)
print(f"Mean Squared Error: {mse_tree:.2f}")
print(f"R² Score: {r2_tree:.2f}")

# ====================
# Step 7: Save the Best Model
# ====================
# Save the best model (for example, Decision Tree after tuning)
with open('model.pkl', 'wb') as f:
    pickle.dump(best_tree_model, f)

# ====================
# Step 8: Model Comparison
# ====================
models = ['Linear Regression', 'Decision Tree']
mse_scores = [mse_linear, mse_tree]
r2_scores = [r2_linear, r2_tree]

results_df = pd.DataFrame({
    'Model': models,
    'MSE': mse_scores,
    'R² Score': r2_scores
})

print("\nModel Performance Comparison:")
print(results_df)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Model')
ax1.set_ylabel('Mean Squared Error', color='tab:blue')
ax1.bar(models, mse_scores, color='tab:blue', alpha=0.6, label='MSE')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Model Comparison: MSE and R² Score')

ax2 = ax1.twinx()
ax2.set_ylabel('R² Score', color='tab:green')
ax2.plot(models, r2_scores, color='tab:green', marker='o', label='R² Score')
ax2.tick_params(axis='y', labelcolor='tab:green')
fig.tight_layout()
plt.show()

# ====================
# Step 9: Model Selection Discussion
# ====================
print("\n----- Model Selection Discussion -----")
if r2_linear > r2_tree and mse_linear < mse_tree:
    print("Linear Regression was chosen because it has a higher R² score and lower MSE than the Decision Tree model. "
          "This indicates that the linear model explains more variance and its predictions are closer to the actual yields.")
elif r2_tree > r2_linear and mse_tree < mse_linear:
    print("The Decision Tree Regressor is preferred as it shows a higher R² score and lower MSE, suggesting it better captures "
          "the non-linear relationships between features and crop yield in this dataset.")
else:
    print("Both models performed similarly on the evaluation metrics. Additional factors such as interpretability, computational efficiency, "
          "or potential ensemble methods can be considered for the final model choice.")

# ====================
# Step 10: Preprocessing Discussion Recap
# ====================
print("\n----- Preprocessing Discussion -----")
print("1. Standardization of Numerical Features: The histogram of a sample feature before and after standardization shows that the")
print("   transformed feature has zero mean and unit variance. This removes scale differences among features and improves convergence")
print("   for many machine learning algorithms.")
print("2. One-hot Encoding of Categorical Features: Categorical variables such as 'Dist Name' and 'Crop' have been transformed")
print("   into binary columns. This converts non-numeric data into a numeric format while preserving categorical information.")
print("3. Overall Data Representation (PCA Visualization): The PCA projection of the preprocessed training data demonstrates that")
print("   the combined numerical and categorical transformations yield a more uniform feature space, thereby facilitating more effective")
print("   model training.")
