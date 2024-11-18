import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#%% Reading and preprocessing data 

data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Fill NaN in Sleep Disorder with None
data.fillna('None',inplace=True)

# Splitting blood pressure into upper and lower blood pressure
data['Systolic'] = data['Blood Pressure'].str.split('/').str[0].astype(int) # upper
data['Diastolic'] = data['Blood Pressure'].str.split('/').str[1].astype(int) # lower

# Dropping blood pressure column (split into two columns) and dropping person id since data is already indexed
data.drop(['Blood Pressure','Person ID'], axis=1, inplace=True)

# Since normal and normal weight seems to be the same category we merge them
data['BMI Category'].replace({'Normal Weight': 'Normal'}, inplace=True)

#%% Data overview

# print(data.head())
# print(data.info())
# print(data.describe())

#%% Class balance

class_counts = data['Sleep Disorder'].value_counts()

plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar')
plt.xlabel('Klasa')
plt.ylabel('Broj primera')
plt.show()

#%% Exploratory Analysis

sns.set(style="white")
sns.set_palette(palette='Set3')

# Extracting numerical variables
num_vars = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# On main diagonal histogram is plotted, otherwise feature dependency is plotted with regression line 
pair_grid = sns.PairGrid(data=data[num_vars], diag_sharey=False)
pair_grid.map_diag(sns.histplot, kde=True)
pair_grid.map_offdiag(sns.regplot, scatter_kws={'s':50, 'alpha':0.5}, line_kws={'color':'red'})

# Correlation matrix of numerical variables
corr_matrix = data[num_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.show()

#%% Data preprocessing

X = data.drop(['Sleep Disorder'], axis=1) 
y = data['Sleep Disorder']

# Label encoding for categorical variables in X
label_encoders = {}  # To store the encoder objects for potential inverse transformations later

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encoding the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

#Applying StandardScaler to Numerical Variables
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_vars] = scaler.fit_transform(X_train[num_vars])
X_test_scaled[num_vars] = scaler.transform(X_test[num_vars])

#%% ML models

# Function to train and validate models
def train_and_cross_validate(model, X_train, y_train, cv=5):

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"{model.__class__.__name__} Cross-Validation F1_weighted: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    model.fit(X_train, y_train)  
    return model


# Function to evaluate and plot models   
def evaluate_and_plot_confusion_matrix(model, X_train, y_train, X_test, y_test, class_names):

    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("\n" + "="*80 + "\n")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=class_names)
    disp.plot()


# Defining class names for confusion matrix display
class_names = ['None', 'Sleep Apnea', 'Insomnia']


# ML models 
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, class_weight='balanced'),
    "SVM": SVC(class_weight='balanced', kernel='linear'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=10),
    "Gradient Boosting" : GradientBoostingClassifier(random_state=10),
}


# Iterate over models, cross-validate, train, predict, and generate classification report and confusion matrix
for i, (name, model) in enumerate(models.items()):
    X_train_set, X_test_set = (X_train_scaled, X_test_scaled) if name in ["Logistic Regression", "SVM"] else (X_train, X_test)
    
    trained_model = train_and_cross_validate(model, X_train_set, y_train)
    evaluate_and_plot_confusion_matrix(trained_model, X_test_set, y_test, X_test_set, y_test, class_names)


#%% Fine tuning - Grid Search

# Function to perform grid search on models
def perform_grid_search(model, params, X_train, y_train):

    grid_search = GridSearchCV(model, params, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Parameters for grid search
param_grids = {
    "Logistic Regression": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'saga']},
    "SVM": {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel':['linear','poly','rbf']},
    "Random Forest":
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
    "Gradient Boosting":
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4]
        }
}



# Model training, hyperparameter tuning, evaluation, and plotting
for i, (name, model) in enumerate(models.items()):

    X_train_set, X_test_set = (X_train_scaled, X_test_scaled) if name in ["Logistic Regression", "SVM"] else (X_train, X_test)
    best_model, best_params = perform_grid_search(model, param_grids[name], X_train_set, y_train)
    evaluate_and_plot_confusion_matrix(best_model, X_test_set, y_test, X_test_set, y_test, class_names)
    print(f"Best Parameters for {name}: {best_params}")
    print("\n" + "="*80 + "\n")


#%% Feature importance

# Training RandomForestClassifier to extract feature importance

rf = RandomForestClassifier(class_weight='balanced', random_state=42,bootstrap= True, max_depth= None, min_samples_leaf=1, min_samples_split=10, n_estimators= 50).fit(X_train, y_train)

feature_importances = rf.feature_importances_

features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

features_df.sort_values(by='Importance', ascending=False).plot(x='Feature',y='Importance',kind='bar')

# Training GradietBoostingClassifier to extract feature importance

gbc = GradientBoostingClassifier(random_state=10, max_depth= 10, min_samples_leaf=2, min_samples_split=5, n_estimators= 100).fit(X_train, y_train)

feature_importances = gbc.feature_importances_

features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

features_df.sort_values(by='Importance', ascending=False).plot(x='Feature',y='Importance',kind='bar')

