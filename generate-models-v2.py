import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pymysql
import joblib

def preprocess_data(data_df):
    # Convert the columns to boolean
    data_df['max_people'] = data_df['max_people'].astype(float)
    data_df['num_rooms'] = data_df['num_rooms'].astype(float)
    data_df['rent_budget'] = data_df['rent_budget'].astype(float)
    data_df['alcohol_consumption'] = data_df['alcohol_int']
    data_df['smoking'] = data_df['smoking_int']
    data_df['has_hall'] = data_df['hasHall_int']
    data_df['loves_cooking'] = data_df['cul_skills_int']

    # Rename columns to match the desired names
    data_df.rename(columns={
        'gender': 'gender',
        'rent_budget': 'rent_budget',
        'alcohol_consumption': 'alcohol',
        'dist_from_uni': 'dist_from_uni',
        'smoking': 'smoking',
        'loves_cooking': 'cul_skills',
        'has_hall': 'hasHall',
        'max_people': 'maxPeople',
        'num_rooms': 'numRooms'
    }, inplace=True)

    # Select only the desired attributes
    filtered_data = data_df[[
        'gender', 'rent_budget', 'alcohol', 'dist_from_uni', 'smoking',
        'cul_skills', 'hasHall', 'maxPeople', 'numRooms'
    ]]

    # Define the weights for each attribute
    attribute_weights = {'gender': 3, "rent_budget": 2, "smoking": 1.5, "alcohol": 1.5}

    # Create scalers
    scaler_dist_from_uni = StandardScaler()
    scaler_rent_budget = StandardScaler()
    scaler = StandardScaler()

    # Scale the data
    filtered_data.loc[:, ['maxPeople', 'numRooms']] = scaler.fit_transform(filtered_data[['maxPeople', 'numRooms']])
    filtered_data.loc[:, 'dist_from_uni'] = scaler_dist_from_uni.fit_transform(filtered_data[['dist_from_uni']])
    filtered_data.loc[:, 'rent_budget'] = scaler_rent_budget.fit_transform(filtered_data[['rent_budget']])

    for attribute, weight in attribute_weights.items():
        filtered_data.loc[:, attribute] = filtered_data[attribute] * weight

    return filtered_data, scaler_dist_from_uni, scaler_rent_budget, scaler


def train_model(X):
    # Train the model
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn.fit(X)
    return knn


def save_model(knn, scaler_dist_from_uni, scaler_rent_budget, scaler):
    # Save the model and scalers
    joblib.dump(knn, 'models/knn_model.joblib')
    joblib.dump(scaler_dist_from_uni, 'models/scaler_dist_from_uni.joblib')
    joblib.dump(scaler_rent_budget, 'models/scaler_rent_budget.joblib')
    joblib.dump(scaler, 'models/scaler_standard.joblib')


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

# Define your SQL query to fetch the data
sql_query = "SELECT *, CAST(alcohol_consumption AS UNSIGNED) AS alcohol_int, CAST(smoking AS UNSIGNED) AS smoking_int, CAST(has_hall AS UNSIGNED) AS hasHall_int, CAST(loves_cooking AS UNSIGNED) AS cul_skills_int FROM roommate_preferences;"

# Execute the SQL query and fetch the data
with conn.cursor() as cursor:
    cursor.execute(sql_query)
    data = cursor.fetchall()

# Close the database connection
conn.close()

# Convert the fetched data into a pandas DataFrame
data_df = pd.DataFrame(data)

# Preprocess the data
filtered_data, scaler_dist_from_uni, scaler_rent_budget, scaler = preprocess_data(data_df)

# Train the model
knn = train_model(filtered_data)

# Save the model
save_model(knn, scaler_dist_from_uni, scaler_rent_budget, scaler)

print("Model trained and saved successfully!")










#
#
#
#
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.neighbors import NearestNeighbors
# import pymysql
#
# # data = pd.read_csv('user_data_clean.csv')
#
#
# # data.head()
#
#
# # Load Spring application properties
# your_host = "localhost"
# your_port = 3306
# your_database_name = "entrepriseProject"
# your_username = "root"
# your_password = "root"
#
# # Establish a connection to the MySQL database
# conn = pymysql.connect(
#     host=your_host,
#     port=your_port,
#     user=your_username,
#     password=your_password,
#     database=your_database_name,
#     charset='utf8mb4',
#     cursorclass=pymysql.cursors.DictCursor
# )
#
#
# # Define your SQL query to fetch the data
# sql_query = "SELECT *, CAST(alcohol_consumption AS UNSIGNED) AS alcohol_int, CAST(smoking AS UNSIGNED) AS smoking_int, CAST(has_hall AS UNSIGNED) AS hasHall_int, CAST(loves_cooking AS UNSIGNED) AS cul_skills_int FROM roommate_preferences;"
#
# # Execute the SQL query and fetch the data
# with conn.cursor() as cursor:
#     cursor.execute(sql_query)
#     data = cursor.fetchall()
#
# # Close the database connection
# conn.close()
#
# # Convert the fetched data into a pandas DataFrame
# data_df = pd.DataFrame(data)
#
#
# # Convert the columns to boolean
# data_df['max_people'] = data_df['max_people'].astype(float)
# data_df['num_rooms'] = data_df['num_rooms'].astype(float)
# data_df['rent_budget'] = data_df['rent_budget'].astype(float)
# data_df['alcohol_consumption'] = data_df['alcohol_int']
# data_df['smoking'] = data_df['smoking_int']
# data_df['has_hall'] = data_df['hasHall_int']
# data_df['loves_cooking'] = data_df['cul_skills_int']
#
#
# print(data_df.head())
#
#
#
# # Rename columns to match the desired names
# # Rename columns to match the desired names
# data_df.rename(columns={
#     'gender': 'gender',
#     'rent_budget': 'rent_budget',
#     'alcohol_consumption': 'alcohol',
#     'dist_from_uni': 'dist_from_uni',
#     'smoking': 'smoking',
#     'loves_cooking': 'cul_skills',
#     'has_hall': 'hasHall',
#     'max_people': 'maxPeople',
#     'num_rooms': 'numRooms'
# }, inplace=True)
#
# print(data_df.head())
#
# #
# # # Select only the desired attributes
# filtered_data = data_df[[
#     'gender', 'rent_budget', 'alcohol', 'dist_from_uni', 'smoking',
#     'cul_skills', 'hasHall', 'maxPeople', 'numRooms'
# ]]
#
#
#
#
#
#
#
# # # Define the weights for each attribute
# attribute_weights = {'gender': 3, "rent_budget":2, "smoking":1.5, "alcohol": 1.5}  # You can adjust these weights based on the importance you want to assign
#
#
#
# # creating the scalers
# filtered_data.head()
#
# scaler_dist_from_uni = StandardScaler()
# scaler_rent_budget = StandardScaler()
# scaler = StandardScaler()
#
#
#
#
# filtered_data.loc[:, ['maxPeople', 'numRooms']] = scaler.fit_transform(filtered_data[['maxPeople', 'numRooms']])
# filtered_data.loc[:, 'dist_from_uni'] = scaler_dist_from_uni.fit_transform(filtered_data[['dist_from_uni']])
# filtered_data.loc[:, 'rent_budget'] = scaler_rent_budget.fit_transform(filtered_data[['rent_budget']])
#
# for attribute, weight in attribute_weights.items():
#     filtered_data.loc[:, attribute] = filtered_data[attribute] * weight
#
# X = filtered_data
#
#
# # training the model
# knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
# knn.fit(X)
#
#
# print(filtered_data.head())
#
#
# # exporting the model
# import joblib
# joblib.dump(knn, '../models/knn_model.joblib')
# joblib.dump(scaler_dist_from_uni, '../models/scaler_dist_from_uni.joblib')
# joblib.dump(scaler_rent_budget, '../models/scaler_rent_budget.joblib')
# joblib.dump(scaler, '../models/scaler_standard.joblib')