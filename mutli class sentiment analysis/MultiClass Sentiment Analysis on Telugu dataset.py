
import pandas as pd

# Load the dataset from the provided Excel file
file_path = '/content/NLP DATA.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
data.head(), data.info()

pip install seaborn

# Check the distribution of sentiment labels
sentiment_distribution = data['Sentiment'].value_counts()

sentiment_distribution

"""Dataset Cleaning & Training Testing sets"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

# Function to clean Telugu text
def clean_text(text):
    # Remove any non-Telugu characters and extra spaces
    text = re.sub('[^\u0C00-\u0C7F]+', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Apply text cleaning
data['cleaned_sentence'] = data['Sentence'].apply(clean_text)

# Encoding the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Sentiment'])

# Features and labels
X = data['cleaned_sentence']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Overview of preprocessing
X_train_tfidf.shape, X_test_tfidf.shape

"""Logisctic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_logistic = logistic_model.predict(X_test_tfidf)

# Evaluate the model
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_report = classification_report(y_test, y_pred_logistic, target_names=label_encoder.classes_)

logistic_accuracy, logistic_report

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate Logistic Regression
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
logistic_report = classification_report(y_test, y_pred_logistic)
logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logistic)

print("Accuracy of Logistic Regression:", logistic_accuracy)
print("Classification Report:\n", logistic_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(logistic_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""Support Vector Machine"""

from sklearn.svm import SVC

# Initialize and train the Support Vector Machine model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_svm = svm_model.predict(X_test_tfidf)

# Evaluate the model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_)

svm_accuracy, svm_report

# Assuming you have predictions as y_pred_svm from an SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_report = classification_report(y_test, y_pred_svm)
svm_confusion_matrix = confusion_matrix(y_test, y_pred_svm)

print("Accuracy of SVM:", svm_accuracy)
print("Classification Report:\n", svm_report)

# Visualize the confusion matrix for SVM
plt.figure(figsize=(8, 6))
sns.heatmap(svm_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate the model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_report = classification_report(y_test, y_pred_nb, target_names=label_encoder.classes_)

nb_accuracy, nb_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate the model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_report = classification_report(y_test, y_pred_nb)
nb_confusion_matrix = confusion_matrix(y_test, y_pred_nb)

print("Accuracy of Naive Bayes:", nb_accuracy)
print("Classification Report:\n", nb_report)

# Visualize the confusion matrix for Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(nb_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

"""Simple Neural Network"""

from sklearn.neural_network import MLPClassifier

# Initialize and train the MLP Classifier
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_mlp = mlp_model.predict(X_test_tfidf)

# Evaluate the model
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_report = classification_report(y_test, y_pred_mlp, target_names=label_encoder.classes_)

mlp_accuracy, mlp_report

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the MLP Classifier
# Here, we use one hidden layer with 100 neurons, which is a good starting point.
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

# Train the model on the training data
mlp_model.fit(X_train_tfidf, y_train)

# Predict on the testing set
y_pred_mlp = mlp_model.predict(X_test_tfidf)

# Evaluate the model
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_report = classification_report(y_test, y_pred_mlp)
mlp_confusion_matrix = confusion_matrix(y_test, y_pred_mlp)

print("Accuracy of MLP Neural Network:", mlp_accuracy)
print("Classification Report:\n", mlp_report)

# Visualize the confusion matrix for the MLP Neural Network
plt.figure(figsize=(8, 6))
sns.heatmap(mlp_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for MLP Neural Network')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns



# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Initialize and train the Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Predict on the testing set
y_pred_decision_tree = decision_tree_model.predict(X_test_tfidf)

# Evaluate the model
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)
decision_tree_report = classification_report(y_test, y_pred_decision_tree, target_names=label_encoder.classes_)
decision_tree_confusion_matrix = confusion_matrix(y_test, y_pred_decision_tree)

print("Accuracy of Decision Tree:", decision_tree_accuracy)
print("Classification Report:\n", decision_tree_report)

# Visualize the confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(decision_tree_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to display performance metrics and visualize the prediction distribution
def evaluate_model(y_true, y_pred, model_name):
    # Generate and print the classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization of the prediction distribution
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate percentages for each label
    total_predictions = cm.sum()
    percentages = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} predictions percentage for {model_name}: {percentages[i] * 100:.2f}%")

# Evaluate Decision Tree
evaluate_model(y_test, y_pred_decision_tree, "Decision Tree")

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the provided Excel file
file_path = '/Users/bharadwajssk/Downloads/NLP DATA.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.info())

# Check the distribution of sentiment labels
sentiment_distribution = data['Sentiment'].value_counts()
print(sentiment_distribution)

# Function to clean Telugu text
def clean_text(text):
    # Remove any non-Telugu characters and extra spaces
    text = re.sub('[^\u0C00-\u0C7F]+', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Apply text cleaning
data['cleaned_sentence'] = data['Sentence'].apply(clean_text)

# Encoding the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Sentiment'])

# Features and labels
X = data['cleaned_sentence']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Check the distribution of sentiment labels after SMOTE
resampled_sentiment_distribution = pd.Series(y_train_resampled).value_counts()
print("Resampled sentiment distribution:\n", resampled_sentiment_distribution)

# Initialize and train the Gradient Boosting model
gradient_boosting_model = GradientBoostingClassifier(random_state=42)
gradient_boosting_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Predict on the testing set
y_pred_gradient_boosting = gradient_boosting_model.predict(X_test_tfidf)

# Evaluate the model
gradient_boosting_accuracy = accuracy_score(y_test, y_pred_gradient_boosting)
gradient_boosting_report = classification_report(y_test, y_pred_gradient_boosting, target_names=label_encoder.classes_)
gradient_boosting_confusion_matrix = confusion_matrix(y_test, y_pred_gradient_boosting)

print("Accuracy of Gradient Boosting:", gradient_boosting_accuracy)
print("Classification Report:\n", gradient_boosting_report)

# Visualize the confusion matrix for Gradient Boosting
plt.figure(figsize=(8, 6))
sns.heatmap(gradient_boosting_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Gradient Boosting')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to display performance metrics and visualize the prediction distribution
def evaluate_model(y_true, y_pred, model_name):
    # Generate and print the classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization of the prediction distribution
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate percentages for each label
    total_predictions = cm.sum()
    percentages = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} predictions percentage for {model_name}: {percentages[i] * 100:.2f}%")

# Evaluate Gradient Boosting
evaluate_model(y_test, y_pred_gradient_boosting, "Gradient Boosting")

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the provided Excel file
file_path = '/Users/bharadwajssk/Downloads/NLP DATA.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.info())

# Check the distribution of sentiment labels
sentiment_distribution = data['Sentiment'].value_counts()
print(sentiment_distribution)

# Function to clean Telugu text
def clean_text(text):
    # Remove any non-Telugu characters and extra spaces
    text = re.sub('[^\u0C00-\u0C7F]+', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Apply text cleaning
data['cleaned_sentence'] = data['Sentence'].apply(clean_text)

# Encoding the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Sentiment'])

# Features and labels
X = data['cleaned_sentence']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Check the distribution of sentiment labels after SMote
resampled_sentiment_distribution = pd.Series(y_train_resampled).value_counts()
print("Resampled sentiment distribution:\n", resampled_sentiment_distribution)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Predict on the testing set
y_pred_random_forest = random_forest_model.predict(X_test_tfidf)

# Evaluate the model
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)
random_forest_report = classification_report(y_test, y_pred_random_forest, target_names=label_encoder.classes_)
random_forest_confusion_matrix = confusion_matrix(y_test, y_pred_random_forest)

print("Accuracy of Random Forest:", random_forest_accuracy)
print("Classification Report:\n", random_forest_report)

# Visualize the confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(random_forest_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to display performance metrics and visualize the prediction distribution
def evaluate_model(y_true, y_pred, model_name):
    # Generate and print the classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization of the prediction distribution
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate percentages for each label
    total_predictions = cm.sum()
    percentages = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} predictions percentage for {model_name}: {percentages[i] * 100:.2f}%")

# Evaluate Random Forest
evaluate_model(y_test, y_pred_random_forest, "Random Forest")

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the provided Excel file
file_path = '/Users/bharadwajssk/Downloads/NLP DATA.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.info())

# Check the distribution of sentiment labels
sentiment_distribution = data['Sentiment'].value_counts()
print(sentiment_distribution)

# Function to clean Telugu text
def clean_text(text):
    # Remove any non-Telugu characters and extra spaces
    text = re.sub('[^\u0C00-\u0C7F]+', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Apply text cleaning
data['cleaned_sentence'] = data['Sentence'].apply(clean_text)

# Encoding the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Sentiment'])

# Features and labels
X = data['cleaned_sentence']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Check the distribution of sentiment labels after SMOTE
resampled_sentiment_distribution = pd.Series(y_train_resampled).value_counts()
print("Resampled sentiment distribution:\n", resampled_sentiment_distribution)

# Initialize and train the AdaBoost model
ada_boost_model = AdaBoostClassifier(random_state=42)
ada_boost_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Predict on the testing set
y_pred_ada_boost = ada_boost_model.predict(X_test_tfidf)

# Evaluate the model
ada_boost_accuracy = accuracy_score(y_test, y_pred_ada_boost)
ada_boost_report = classification_report(y_test, y_pred_ada_boost, target_names=label_encoder.classes_)
ada_boost_confusion_matrix = confusion_matrix(y_test, y_pred_ada_boost)

print("Accuracy of AdaBoost:", ada_boost_accuracy)
print("Classification Report:\n", ada_boost_report)

# Visualize the confusion matrix for AdaBoost
plt.figure(figsize=(8, 6))
sns.heatmap(ada_boost_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for AdaBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to display performance metrics and visualize the prediction distribution
def evaluate_model(y_true, y_pred, model_name):
    # Generate and print the classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization of the prediction distribution
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate percentages for each label
    total_predictions = cm.sum()
    percentages = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} predictions percentage for {model_name}: {percentages[i] * 100:.2f}%")

# Evaluate AdaBoost
evaluate_model(y_test, y_pred_ada_boost, "AdaBoost")

pip install xgboost

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the provided Excel file
file_path = '/Users/bharadwajssk/Downloads/NLP DATA.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
print(data.head())
print(data.info())

# Check the distribution of sentiment labels
sentiment_distribution = data['Sentiment'].value_counts()
print(sentiment_distribution)

# Function to clean Telugu text
def clean_text(text):
    # Remove any non-Telugu characters and extra spaces
    text = re.sub('[^\u0C00-\u0C7F]+', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

# Apply text cleaning
data['cleaned_sentence'] = data['Sentence'].apply(clean_text)

# Encoding the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['Sentiment'])

# Features and labels
X = data['cleaned_sentence']
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_tfidf_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Check the distribution of sentiment labels after SMOTE
resampled_sentiment_distribution = pd.Series(y_train_resampled).value_counts()
print("Resampled sentiment distribution:\n", resampled_sentiment_distribution)

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train_tfidf_resampled, y_train_resampled)

# Predict on the testing set
y_pred_xgb = xgb_model.predict(X_test_tfidf)

# Evaluate the model
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_report = classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_)
xgb_confusion_matrix = confusion_matrix(y_test, y_pred_xgb)

print("Accuracy of XGBoost:", xgb_accuracy)
print("Classification Report:\n", xgb_report)

# Visualize the confusion matrix for XGBoost
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix for XGBoost')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to display performance metrics and visualize the prediction distribution
def evaluate_model(y_true, y_pred, model_name):
    # Generate and print the classification report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(f"Classification Report for {model_name}:\n{report}")

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}%\n")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Visualization of the prediction distribution
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate percentages for each label
    total_predictions = cm.sum()
    percentages = cm.diagonal() / cm.sum(axis=1)
    for i, label in enumerate(label_encoder.classes_):
        print(f"{label} predictions percentage for {model_name}: {percentages[i] * 100:.2f}%")

# Evaluate XGBoost
evaluate_model(y_test, y_pred_xgb, "XGBoost")

