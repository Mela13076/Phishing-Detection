import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Load the datasets
train_data = pd.read_csv('urls_train2.csv')
test_data = pd.read_csv('urls_test2.csv')


# Identify categorical and numerical features
categorical_features = ['domain']

#first round of feature extraction testing
#numerical_features = ['path_length', 'use_https', 'num_subdomains', 'token_count_path', 'contains_suspicious_keywords']

#Second round of feature extraction testing
numerical_features = ['path_length', 'use_https', 'num_subdomains', 'token_count_path', 'contains_suspicious_keywords',
                      'url_length', 'query_count', 'is_shortened', 'has_at_symbol', 'count_hyphens']

# Preprocessing for numerical data: fill missing values and scale
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())  # Scale data
])

# Preprocessing for categorical data: fill missing values and apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Convert categorical data
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

# Create and configure pipeline with SMOTE
rf_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                 ('smote', SMOTE(random_state=42)),
                                 ('classifier', rf_model)])

dt_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                 ('smote', SMOTE(random_state=42)),
                                 ('classifier', dt_model)])

knn_pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                                  ('smote', SMOTE(random_state=42)),
                                  ('classifier', knn_model)])

# Prepare target variables
X_train = train_data.drop(['label'], axis=1)
y_train = train_data['label']
X_test = test_data.drop(['label'], axis=1)
y_test = test_data['label']

# Fit the pipelines to train the models
rf_pipeline.fit(X_train, y_train)
dt_pipeline.fit(X_train, y_train)
knn_pipeline.fit(X_train, y_train)

# Convert y_test to binary 
# phishing is coded as 1 and legitimate as 0 , used for the ROC-AUC
y_test_binary = label_binarize(y_test, classes=['legitimate', 'phishing']).ravel()

# Predict on the test set and evaluate
models = {'Random Forest': rf_pipeline, 'Decision Tree': dt_pipeline, 'KNN': knn_pipeline}
for name, pipeline in models.items():
    print(f"Evaluation of {name}")
    y_pred = pipeline.predict(X_test)
    #getting probabilities for the positve class
    y_pred_proba = pipeline.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))
    roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
    #roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovo', average='weighted')
    print("ROC-AUC Score:", roc_auc)
    print("\n")

# cross-validation to assess model reliability for Random Forest
print("Cross-validation Scores for Random Forest:")
scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validated Scores RF:", scores)

# Cross-validation for Decision Tree
print("Cross-validation Scores for Decision Tree:")
dt_scores = cross_val_score(dt_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validated Scores DT:", dt_scores)

# Cross-validation for KNN
print("Cross-validation Scores for KNN:")
knn_scores = cross_val_score(knn_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validated Scores KNN:", knn_scores)

