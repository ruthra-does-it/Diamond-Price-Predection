# Step 1 — Data collection and initial setup

import pandas as pd

df = pd.read_csv("/Users/rudyman/desktop/diamond/diamonds.csv")

# 2) Quick structure checks
print("Shape:", df.shape)                
print("\nColumns:", list(df.columns))     

print("\nData types:")
print(df.dtypes)

print("\nHead (first 5 rows):")
print(df.head())

# 3) Missing values
print("\nMissing values per column:")
print(df.isna().sum())
# Step 2 — Data Preprocessing

import numpy as np

# 1) Handle missing values
print("Missing values before cleaning:")
print(df.isna().sum())
# 2) Replace invalid values (x, y, z cannot be zero)
for col in ['x', 'y', 'z']:
    df[col] = df[col].replace(0, np.nan)

# Check again
print("\nMissing values after replacing zeros with NaN:")
print(df.isna().sum())
# 3) Impute missing values (simple strategy: fill with median of each column)
df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing values after imputation:")
print(df.isna().sum())
# 4) Confirm dataset shape remains same
print("\nFinal shape:", df.shape)
# Step 3 — Outlier Handling

import matplotlib.pyplot as plt
import seaborn as sns

# Numerical columns to check
num_cols = ['carat', 'price', 'x', 'y', 'z']

# Function to remove outliers using IQR
def remove_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Visualize before cleaning
plt.figure(figsize=(12,6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f"Before Cleaning: {col}")
plt.tight_layout()
plt.show()

# Apply outlier removal for each numeric column
for col in num_cols:
    df = remove_outliers(df, col)

# Visualize after cleaning
plt.figure(figsize=(12,6))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f"After Cleaning: {col}")
plt.tight_layout()
plt.show()

print("Shape after outlier removal:", df.shape)
# Step 4 — Skewness Handling

# 1) Check skewness
print("Skewness before transformation:")
print(df[['carat', 'price', 'x', 'y', 'z']].skew())

# 2) Apply log transformation to skewed features
import numpy as np

# Use log1p (log(1 + x)) to avoid issues with zero values
df['carat_log'] = np.log1p(df['carat'])
df['price_log'] = np.log1p(df['price'])
df['x_log'] = np.log1p(df['x'])
df['y_log'] = np.log1p(df['y'])
df['z_log'] = np.log1p(df['z'])

# 3) Check skewness after transformation
print("\nSkewness after log transformation:")
print(df[['carat_log', 'price_log', 'x_log', 'y_log', 'z_log']].skew())
# Step 5 — EDA: Distribution plots

num_cols = ['price', 'carat', 'x', 'y', 'z']
plt.figure(figsize=(12,8))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()
# Count plots for categorical features
cat_cols = ['cut', 'color', 'clarity']
plt.figure(figsize=(12,4))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 3, i)
    sns.countplot(x=df[col], order=sorted(df[col].unique()))
    plt.title(f"Count of {col}")
plt.tight_layout()
plt.show()
# Boxplots: Price vs categorical features
plt.figure(figsize=(15,5))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df[col], y=df['price'])
    plt.title(f"Price vs {col}")
plt.tight_layout()
plt.show()
# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[['carat','x','y','z','price']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# Pairplot for carat, x, y, z, price
sns.pairplot(df[['carat','x','y','z','price']])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()
# Average price per category
plt.figure(figsize=(15,4))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 3, i)
    avg_price = df.groupby(col)['price'].mean().sort_values()
    sns.barplot(x=avg_price.index, y=avg_price.values)
    plt.title(f"Average Price by {col}")
plt.tight_layout()
plt.show()
# Step 6 — Feature Engineering

# 1) Convert price to INR (assume 1 USD = 83 INR)
df['price_inr'] = df['price'] * 83

# 2) Create new features
df['volume'] = df['x'] * df['y'] * df['z']
df['price_per_carat'] = df['price'] / df['carat']
df['dimension_ratio'] = (df['x'] + df['y']) / (2 * df['z'])

# 3) Categorize carat
def carat_category(carat):
    if carat < 0.5:
        return 'Light'
    elif carat <= 1.5:
        return 'Medium'
    else:
        return 'Heavy'

df['carat_category'] = df['carat'].apply(carat_category)

# 4) Preview new columns
print(df[['price', 'price_inr', 'volume', 'price_per_carat', 'dimension_ratio', 'carat_category']].head())
# Step 7 — Feature Selection

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1) Correlation matrix for numeric features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df[['carat','x','y','z','depth','table','volume','price_inr']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# 2) Feature Importance using Random Forest

# Define features (drop target 'price_inr')
X = df.drop(columns=['price_inr','price','price_log'])
y = df['price_inr']

# Encode categorical variables (simple label encoding for now)
from sklearn.preprocessing import LabelEncoder
for col in ['cut','color','clarity','carat_category']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)

print(feat_imp)

# Plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp['Importance'], y=feat_imp['Feature'])
plt.title("Feature Importance (Random Forest)")
plt.show()
# Step 8 — Encoding categorical features

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# 1) Define ordered categories
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J','I','H','G','F','E','D']
clarity_order = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

# 2) Apply Ordinal Encoding
ord_enc = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
df[['cut_enc','color_enc','clarity_enc']] = ord_enc.fit_transform(df[['cut','color','clarity']])

# 3) Label Encoding for carat_category
le = LabelEncoder()
df['carat_cat_enc'] = le.fit_transform(df['carat_category'])

# 4) Preview encoded columns
print(df[['cut','cut_enc','color','color_enc','clarity','clarity_enc','carat_category','carat_cat_enc']].head())
# Step 9 — Regression Models

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import numpy as np

# Features (drop target and original price columns)
X = df[['carat','x','y','z','depth','table','cut_enc','color_enc','clarity_enc','carat_cat_enc','volume','dimension_ratio']]
y = df['price_inr']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

results = {}
for name, model in models.items():
    mae, mse, rmse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)
# ANN Regression Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build ANN
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train ANN
history = ann.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate ANN
y_pred_ann = ann.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred_ann)
mse = mean_squared_error(y_test, y_pred_ann)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_ann)

print("\nANN Performance:")
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
# Save the best regression model (Random Forest) from your models dictionary

import joblib

# Save the model using the key name from your dictionary
joblib.dump(models["Random Forest"], "best_price_model.pkl")

print("✅ Random Forest model saved as best_price_model.pkl")
# Step 10 — Clustering Setup

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1) Drop price columns
df_cluster = df.drop(columns=['price', 'price_inr', 'price_log'])

# 2) Select features for clustering
features = ['carat','x','y','z','depth','table','cut_enc','color_enc','clarity_enc','carat_cat_enc','volume','dimension_ratio']
X_cluster = df_cluster[features]

# 3) Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 4) Elbow Method to find optimal clusters
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()
# Final K-Means clustering with K = 3
kmeans_final = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

# Save the clustering model
import joblib
joblib.dump(kmeans_final, "best_cluster_model.pkl")
print("✅ Clustering model saved as best_cluster_model.pkl")
# Cluster summary
cluster_summary = df.groupby('cluster').agg({
    'carat': 'mean',
    'price_inr': 'mean',
    'cut': lambda x: x.value_counts().idxmax()
}).rename(columns={'cut': 'most_common_cut'})

print(cluster_summary)
# Map cluster names
cluster_names = {
    0: "Affordable Small Diamonds",
    1: "Mid-range Balanced Diamonds",
    2: "Premium Heavy Diamonds"
}

df['cluster_name'] = df['cluster'].map(cluster_names)
print(df[['cluster', 'cluster_name']].head(100))