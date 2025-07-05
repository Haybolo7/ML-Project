import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# ---------------------
# DATA PRE-PROCESSING
# ---------------------
def preprocess_data(df):
    # Removing irrelevant column
    if 'Loan_ID' in df.columns:
        df=df.drop('Loan_ID', axis=1)

    # handling missing data values
    df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married']=df['Married'].fillna(df['Married'].mode()[0])
    df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Dependents']=df['Dependents'].str.replace('+', ' ', regex=False)
    df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])

    # handling categorical values
    binary_cols = ['Gender', 'Married', 'Self_Employed']
    le = LabelEncoder()
    # Label encoding binary features
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # OneHot encoding multi-class features
    df = pd.get_dummies(df, columns=['Education', 'Property_Area'], drop_first=True)
    # Encoding target variable
    df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

    # Spliting data into X and y
    X=df.drop('Loan_Status', axis=1)
    y=df['Loan_Status']

    # Feature Scaling the data
    scaler=StandardScaler()
    X_scaled= scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Balancing dataset using SMOTETomek
    smk=SMOTETomek(random_state=42)
    X_resampled, y_resampled = smk.fit_resample(X_scaled,y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name='Loan_Status')

    return X_resampled, y_resampled, scaler


#--------------------------------
# MODEL TRAINING ABD EVALUATION
# -------------------------------
def train_models(X_train, X_test, y_train, y_test, scaler):
    # Hyper-parameter Tuning the models
    models = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    results = []

    for name, config in models.items():
        print(f"\nTraining {name}...")

        rs = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=20,
            scoring='f1',
            cv=5,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )

        rs.fit(X_train, y_train)
        best_model = rs.best_estimator_


        # Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

        print(f"\nModel: {name}")
        print(f"Best Params: {rs.best_params_}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}")
        print(f"Cross-Validation Std Dev: {cv_scores.std():.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("-" * 60)

        results.append({
            'name': name,
            'model': best_model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'f1': f1,
            'accuracy':acc
        })

    model_selection(results, scaler)


# -----------------
# MODEL SELECTION
# -----------------
def model_selection(results, scaler):
    # Selecting model with highest Test F1 Score
    best_result = max(results, key=lambda x: x['f1'])
    best_model = best_result["model"]
    best_model_name = best_result["name"]
    
    print(f"\nBest Model Based on Test F1 Score: {best_model_name}")
    print(f"Test F1 Score: {best_result['f1']:.4f}")
    print(f"Test Accuracy: {best_result['accuracy']:.4f}")
    print(f"CV Accuracy: {best_result['cv_mean']:.4f}")
    print(f"CV Std Dev: {best_result['cv_std']:.4f}")

    # Saving best model
    model_name = "final_model.pkl"
    with open(f"model/{model_name}", 'wb') as f:
        pickle.dump(best_model, f)

    # Saving the scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("***** Model and Scaler saved *****")

     # Feature importance (for tree-based models only)
    if hasattr(best_model, 'feature_importances_'):
        import matplotlib.pyplot as plt

        importances = best_model.feature_importances_
        features = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History',
            'Property_Area_Semiurban', 'Property_Area_Urban'
        ]

        plt.figure(figsize=(10, 5))
        plt.barh(features, importances)
        plt.xlabel("Feature Importance")
        plt.title(f"Feature Importance in {best_model_name}")
        plt.tight_layout()
        plt.show()


# ----------------
# MAIN EXECUTION
# ----------------
if __name__ == "__main__":
    # Reading the data from csv file
    data = pd.read_csv('data/loan_prediction.csv')

    # Data pre-processing
    X, y, scaler = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model Building
    train_models(X_train, X_test, y_train, y_test, scaler)
