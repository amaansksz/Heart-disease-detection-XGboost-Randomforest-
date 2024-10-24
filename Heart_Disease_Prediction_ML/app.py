from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart.csv')

# Preprocessing
X = data.drop('target', axis=1)  # Feature columns
y = data['target']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
xgb_model = XGBClassifier(eval_metric='logloss')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a Voting Classifier
voting_clf = VotingClassifier(estimators=[('xgb', xgb_model), ('rf', rf_model)], voting='soft')

# Cross-Validation (K-Fold)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=kf, scoring='accuracy')

# Train the Voting Classifier
voting_clf.fit(X_train, y_train)

# Test Accuracy
y_pred = voting_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gather input from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])

    # Prepare the input features in the same order as the training data
    input_features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # Make prediction using custom threshold (e.g., 0.4 instead of 0.5)
    probabilities = voting_clf.predict_proba(input_features)
    heart_disease_prob = probabilities[0][1]  # Probability for class 1 (heart disease)

    # Use a custom threshold for classification
    if heart_disease_prob > 0.4:  # Lowering the threshold from 0.5 to 0.4
        prediction = 1  # Heart disease
    else:
        prediction = 0  # No heart disease

    # Output result
    if prediction == 1:
        result = 'Heart Disease'
    else:
        result = 'No Heart Disease'

    return render_template('result.html', result=result, chosen_model_acc=round(accuracy * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
