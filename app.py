import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from flask import Flask, request, render_template

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv('diabetes.csv')

# Data Preprocessing
X = dataset.iloc[:, :-1].values  # Features
Y = dataset.iloc[:, -1].values   # Target

sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])

# Data Modeling - Decision Tree
dt_classifier = DecisionTreeClassifier(random_state=42)  # Create DecisionTreeClassifier instance
dt_classifier.fit(X_train, Y_train)  # Fit the DecisionTreeClassifier to training data
joblib.dump(dt_classifier, 'Decision_Tree_Classifier.pkl')

# Data Modeling - Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Create RandomForestClassifier instance
rf_classifier.fit(X_train, Y_train)  # Fit the RandomForestClassifier to training data
joblib.dump(rf_classifier, 'Random_Forest_Classifier.pkl')

# Data Modeling - SVM
svm_classifier = SVC(kernel='linear', random_state=42, probability=True)  # Create SVC instance with probability=True
svm_classifier.fit(X_train, Y_train)  # Fit the SVC to training data
joblib.dump(svm_classifier, 'SVM.pkl')

# Data Modeling - Logistic Regression
logistic_classifier = LogisticRegression(random_state=42)  # Create Logistic Regression instance
logistic_classifier.fit(X_train, Y_train)  # Fit the Logistic Regression to training data
joblib.dump(logistic_classifier, 'Logistic_Regression.pkl')

# Load the models
decision_tree_model = joblib.load('Decision_Tree_Classifier.pkl')
random_forest_model = joblib.load('Random_Forest_Classifier.pkl')
svm_model = joblib.load('SVM.pkl')
logistic_regression_model = joblib.load('Logistic_Regression.pkl')

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model = request.form['model']
        float_features = [float(x) for x in request.form.values() if x != selected_model]  # Exclude the selected model value
        final_features_scaled = sc.transform([float_features])

        if selected_model == 'Decision Tree':
            model = decision_tree_model
        elif selected_model == 'Random Forest':
            model = random_forest_model
        elif selected_model == 'SVM':
            model = svm_model
        elif selected_model == 'Logistic Regression':
            model = logistic_regression_model
        else:
            return render_template('index.html', prediction_text="Please select a valid model.")

        prediction = model.predict(final_features_scaled)
        prediction_score = model.predict_proba(final_features_scaled)

        if prediction[0] == 1:
            pred_text = "You have Diabetes, please consult a Doctor."
        elif prediction[0] == 0:
            pred_text = "You don't have Diabetes."
        
        pred_text_0 = f"Probability of class 0: {prediction_score[0][0]}"
        pred_text_1 = f"Probability of class 1: {prediction_score[0][1]}"

        
        return render_template('index.html', prediction_text=pred_text, pred_text_0=pred_text_0, pred_text_1=pred_text_1)

    except Exception as e:
        return render_template('index.html', prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
