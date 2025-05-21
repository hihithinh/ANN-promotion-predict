# %% [markdown]
# # Predicting Employee Promotion with Random Forest

# %% [markdown]
# ## 1. Import Libraries

# %% [code]
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## 2. Load Data

# %% [code]
# Load the training data
try:
    train_df = pd.read_csv('data/train.csv')
    print("Training data loaded successfully.")
    print("Shape of training data:", train_df.shape)
    # print("First 5 rows:")
    # print(train_df.head())
except FileNotFoundError:
    print("Error: 'data/train.csv' not found. Make sure the file is in the correct directory.")
    train_df = None 
except Exception as e:
    print(f"An error occurred: {e}")
    train_df = None

# %% [markdown]
# ## 3. Data Preprocessing

# %% [code]
if train_df is not None:
    # print("\nData Info:")
    # train_df.info()
    # print("\nMissing values per column:")
    # print(train_df.isnull().sum())

    # Drop employee_id and separate features (X) and target (y)
    X = train_df.drop(['employee_id', 'is_promoted'], axis=1)
    y = train_df['is_promoted'].values # Keep y as a 1D array for RandomForestClassifier

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    print(f"\nCategorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Kept for consistency with ANN, though RF is less sensitive
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nShape of X_train before processing: {X_train.shape}")
    print(f"Shape of X_val before processing: {X_val.shape}")

    # Apply the preprocessing pipeline
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    # y_train and y_val are already 1D, which is suitable for RandomForestClassifier

    print(f"\nShape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of X_val_processed: {X_val_processed.shape}")
    print(f"Shape of y_val: {y_val.shape}")

else:
    print("train_df was not loaded. Skipping preprocessing.")
    X_train_processed, X_val_processed, y_train, y_val = None, None, None, None

# %% [markdown]
# ## 4. Model Training (Random Forest)

# %% [code]
rf_model = None
if X_train_processed is not None and y_train is not None:
    print("\nTraining Random Forest model...")
    # Initialize RandomForestClassifier, using class_weight='balanced'
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') 
    
    # Train the model on the original preprocessed training data
    rf_model.fit(X_train_processed, y_train)
    print("Random Forest model trained successfully.")
else:
    print("Skipping Random Forest model training as data is not available.")

# %% [markdown]
# ## 5. Model Evaluation (Random Forest)

# %% [code]
if rf_model is not None and X_train_processed is not None and X_val_processed is not None:
    # Predictions on training data (original training data)
    print("\n--- Evaluation on Training Data (Random Forest - Original) ---")
    y_pred_train_rf = rf_model.predict(X_train_processed)
    train_accuracy_rf = accuracy_score(y_train, y_pred_train_rf)
    print(f"Training Accuracy: {train_accuracy_rf:.4f}")
    print("Training Classification Report:")
    print(classification_report(y_train, y_pred_train_rf))
    print("Training Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred_train_rf))

    # Predictions on validation data (original validation data)
    print("\n--- Evaluation on Validation Data (Random Forest - Original) ---")
    y_pred_val_rf = rf_model.predict(X_val_processed)
    val_accuracy_rf = accuracy_score(y_val, y_pred_val_rf)
    print(f"Validation Accuracy: {val_accuracy_rf:.4f}")
    print("Validation Classification Report:")
    try:
        print(classification_report(y_val, y_pred_val_rf))
        print("\nValidation Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred_val_rf))
    except Exception as e:
        print(f"Could not generate classification report/confusion matrix: {e}")

    # Feature Importances (Optional, but common for RF)
    # if hasattr(rf_model, 'feature_importances_'):
    #     try:
    #         # Get feature names after one-hot encoding
    #         # This requires access to the preprocessor's fitted transformers
    #         feature_names = preprocessor.get_feature_names_out()
    #         importances = rf_model.feature_importances_
    #         feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    #         feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    #         print("\nTop 10 Feature Importances:")
    #         print(feature_importance_df.head(10))
            
    #         # Plot feature importances
    #         plt.figure(figsize=(10, 8))
    #         sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10), palette='viridis')
    #         plt.title('Top 10 Feature Importances - Random Forest')
    #         plt.tight_layout()
    #         plt.show()
    #     except Exception as e:
    #         print(f"Could not display feature importances: {e}")
else:
    print("Skipping Random Forest model evaluation as the model was not trained or data is unavailable.")
