
from pymongo import MongoClient

# Replace 'your_connection_string' with your MongoDB Atlas connection string
client = MongoClient('mongodb://localhost:27017')

# Access your database and collections
db = client['Endsem_T9']
collection_name = ['fersym_T9']

for collection_name in collection_name:
    collection = db[collection_name]
    cursor = collection.find({})
    
    data = list(cursor)
    print(f"Data from collection '{collection_name}':")
    print(data)


# In[5]:


#fertility dataset
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')  
db = client['Endsem_T9']  
collection_name = 'fersym_T9'  # Replace 'your_collection' with your collection name

# Access the collection
collection = db[collection_name]

# Fetch data from the collection
cursor = collection.find({})

# Convert MongoDB cursor to Pandas DataFrame
df = pd.DataFrame(list(cursor))
# Display the DataFrame
print(df)


# In[6]:


# Display the first few rows of the DataFrame
print(df.head())


# In[6]:


#1.SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']  # Replace 'your_database' with your database name
collection_name = 'fersym_T9'  # Replace 'fertility' with your collection name
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

print("\nEncoded Features:")
print(pd.DataFrame(X_encoded, columns=encoded_columns).head())

encoded_columns = []
for col, categories in zip(categorical_cols, encoder.categories_):
    encoded_columns.extend([f"{col}_{category}" for category in categories])
    
# Drop the original categorical columns from X and concatenate encoded columns
X.drop(categorical_cols, axis=1, inplace=True)
X = pd.concat([X, pd.DataFrame(X_encoded, columns=encoded_columns)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train SVC model
svc = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svc.fit(X_train, y_train)

# Model evaluation (if needed)
accuracy = svc.score(X_test, y_test)
print(f"Accuracy: {accuracy}")


# In[17]:


#2.Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']  # Replace 'your_database' with your database name
collection_name = 'fersym_T9'  # Replace 'fertility' with your collection name
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

print("\nEncoded Features:")
print(pd.DataFrame(X_encoded, columns=encoded_columns).head())

encoded_columns = []
for col, categories in zip(categorical_cols, encoder.categories_):
    encoded_columns.extend([f"{col}_{category}" for category in categories])
    
# Drop the original categorical columns from X and concatenate encoded columns
X.drop(categorical_cols, axis=1, inplace=True)
X = pd.concat([X, pd.DataFrame(X_encoded, columns=encoded_columns)], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train Naive Bayes model (GaussianNB for continuous features)
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predictions on test set
predictions = nb.predict(X_test)

# Model evaluation (if needed)
accuracy = (predictions == y_test).sum() / len(y_test)
print(f"Accuracy: {accuracy}")


# In[7]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values with mean imputation
    ('scaler', StandardScaler()),  # Scale features
    ('nb', GaussianNB())  # Naive Bayes classifier
])

# Define hyperparameters for GridSearchCV
param_grid = {
    # Define hyperparameters to optimize (e.g., var_smoothing for GaussianNB)
    'nb__var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]
}

# Perform GridSearchCV for hyperparameter tuning
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid.best_estimator_

# Evaluate the best model on the test set
accuracy = best_model.score(X_test, y_test)
print(f"Improved Accuracy: {accuracy}")


# In[10]:


#3.Isolation Forest - Anomaly Detection Algorithm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']
collection_name = 'fersym_T9'
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

encoded_columns = []  # Define encoded_columns here
for col, categories in zip(categorical_cols, encoder.categories_):
    encoded_columns.extend([f"{col}_{category}" for category in categories])
    
# Drop the original categorical columns from X and concatenate encoded columns
X.drop(categorical_cols, axis=1, inplace=True)
X = pd.concat([X, pd.DataFrame(X_encoded, columns=encoded_columns)], axis=1)

isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the Isolation Forest model
isolation_forest.fit(X)

# Predict anomalies (outliers)
anomaly_scores = isolation_forest.decision_function(X)
anomaly_labels = isolation_forest.predict(X)

# Anomaly scores: The lower the score, the more anomalous the data point
print("Anomaly Scores:")
print(anomaly_scores)

# Anomaly labels: -1 for anomalies/outliers, 1 for inliers
print("Anomaly Labels:")
print(anomaly_labels)

anomaly_count = (anomaly_labels == -1).sum()

# Calculate accuracy
total_samples = len(X)
accuracy = (total_samples - anomaly_count) / total_samples

print("Accuracy of Isolation Forest:", accuracy)


# In[2]:


get_ipython().system('pip install xgboost')


# In[37]:


#4.Gradient Boosting Machine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']  # Replace 'your_database' with your database name
collection_name = 'fersym_T9'  # Replace 'fertility' with your collection name
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

print("\nEncoded Features:")
print(pd.DataFrame(X_encoded, columns=encoded_columns).head())

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column contains strings (categorical)
        df[col] = label_encoder.fit_transform(df[col])

# Separating features and target variable
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[38]:


#5. Ensemble Learning - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']  # Replace 'your_database' with your database name
collection_name = 'fersym_T9'  # Replace 'fertility' with your collection name
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

print("\nEncoded Features:")
print(pd.DataFrame(X_encoded, columns=encoded_columns).head())

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column contains strings (categorical)
        df[col] = label_encoder.fit_transform(df[col])

# Separating features and target variable
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[9]:


#Isolation Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client['Endsem_T9']
collection_name = 'fersym_T9'
collection = db[collection_name]

# Fetch data from MongoDB collection
cursor = collection.find({}, {'Season': 1, 'Childish diseases': 1, 'Accident or serious trauma': 1,
                               'Surgical intervention': 1, 'High fevers in the last year': 1,
                               'Frequency of alcohol consumption': 1, 'Smoking habit': 1,
                               'Diagnosis': 1, 'Age': 1, 'Number of hours spent sitting per day': 1,
                               '_id': 0})
df = pd.DataFrame(list(cursor))

print("Original Data:")
print(df.head())

# Separate features and target class
X = df.drop('Diagnosis', axis=1)  # Features
y = df['Diagnosis']  # Target class

print("\nFeatures (X):")
print(X.head()) 

# Identify categorical columns for encoding
categorical_cols = ['Season', 'Childish diseases', 'Accident or serious trauma', 'Surgical intervention',
                    'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])

encoded_columns = []  # Define encoded_columns here
for col, categories in zip(categorical_cols, encoder.categories_):
    encoded_columns.extend([f"{col}_{category}" for category in categories])
    
# Drop the original categorical columns from X and concatenate encoded columns
X.drop(categorical_cols, axis=1, inplace=True)
X = pd.concat([X, pd.DataFrame(X_encoded, columns=encoded_columns)], axis=1)

isolation_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the Isolation Forest model
isolation_forest.fit(X)

# Predict anomalies (outliers)
anomaly_scores = isolation_forest.decision_function(X)
anomaly_labels = isolation_forest.predict(X)

# Anomaly scores: The lower the score, the more anomalous the data point
print("Anomaly Scores:")
print(anomaly_scores)

# Anomaly labels: -1 for anomalies/outliers, 1 for inliers
print("Anomaly Labels:")
print(anomaly_labels)

anomaly_count = (anomaly_labels == -1).sum()

# Calculate accuracy
total_samples = len(X)
accuracy = (total_samples - anomaly_count) / total_samples

print("Accuracy of Isolation Forest:", accuracy)







