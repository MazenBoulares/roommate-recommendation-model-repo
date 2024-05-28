import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pymysql

# data = pd.read_csv('user_data_clean.csv')


# data.head()


# Load Spring application properties
your_host = "localhost"
your_port = 3306
your_database_name = "entrepriseProject"
your_username = "root"
your_password = "root"

# Establish a connection to the MySQL database
conn = pymysql.connect(
    host=your_host,
    port=your_port,
    user=your_username,
    password=your_password,
    database=your_database_name,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)





categorical_features = ['gender', 'rent_budget', 'alcohol', 'rent_budget', 'dist_from_uni', 'smoking', 'food_pref', 'cul_skills', 'bhk_1',	'bhk_2',	'bhk_3',	'bhk_4']

# Define the weights for each attribute
attribute_weights = {'gender': 3, "rent_budget":1.5, "smoking":1.5, "alcohol": 1.5}  # You can adjust these weights based on the importance you want to assign

filtered_data = data[categorical_features].copy()


filtered_data.head()

scaler_dist_from_uni = StandardScaler()
scaler_rent_budget = StandardScaler()

# Scale 'dist_from_uni' and 'rent_budget' separately
filtered_data['dist_from_uni'] = scaler_dist_from_uni.fit_transform(filtered_data[['dist_from_uni']])
filtered_data['rent_budget'] = scaler_rent_budget.fit_transform(filtered_data[['rent_budget']])

# Apply weights to selected attributes
for attribute, weight in attribute_weights.items():
    filtered_data[attribute] = filtered_data[attribute] * weight

X = filtered_data

knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X)

import joblib
joblib.dump(knn, '../models/knn_model.joblib')
joblib.dump(scaler_dist_from_uni, '../models/scaler_dist_from_uni.joblib')
joblib.dump(scaler_rent_budget, '../models/scaler_rent_budget.joblib')