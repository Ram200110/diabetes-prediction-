#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

def load_data():
    dataset = pd.read_csv('diabetes.csv')
    X = dataset.iloc[:, :-1]  # Selecting all columns except the last one (Outcome)
    Y = dataset.iloc[:, -1]   # Selecting only the last column (Outcome)
    sc = MinMaxScaler(feature_range=(0, 1))
    X_scaled = sc.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=42, stratify=dataset['Outcome'])
    return X_train, X_test, Y_train, Y_test

def train_model(model_name):
    X_train, _, Y_train, _ = load_data()
    model_file = model_name + '.pkl'
    if model_name == 'Decision_Tree_Classifier':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
    elif model_name == 'SVM':
        from sklearn.svm import SVC
        model = SVC(kernel='linear', random_state=42)
    elif model_name == 'Random_Forest_Classifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Invalid model name. Supported models: Decision_Tree_Classifier, logistic_regression, SVM, Random_Forest_Classifier")
    model.fit(X_train, Y_train)
    pickle.dump(model, open(model_file, 'wb'))
    print(f"{model_name} model trained and saved as {model_file}")

def test_model(model_name, model_file):
    _, X_test, _, Y_test = load_data()
    model = pickle.load(open(model_file, 'rb'))
    accuracy = model.score(X_test, Y_test)
    print(f"Accuracy of {model_name} on test set: {accuracy}")

if __name__ == "__main__":
    models = [
        {'name': 'Decision_Tree_Classifier', 'file': 'Decision_Tree_Classifier.pkl'},
        {'name': 'logistic_regression', 'file': 'logistic_regression.pkl'},
        {'name': 'SVM', 'file': 'SVM.pkl'},
        {'name': 'Random_Forest_Classifier', 'file': 'Random_Forest_Classifier.pkl'}
    ]
    for model_info in models:
        train_model(model_info['name'])
        test_model(model_info['name'], model_info['file'])
