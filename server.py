import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import pymysql
import json

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load models and scalers
knn = joblib.load('models/knn_model.joblib')
scaler_dist_from_uni = joblib.load('models/scaler_dist_from_uni.joblib')
scaler_rent_budget = joblib.load('models/scaler_rent_budget.joblib')
scaler_standard = joblib.load('models/scaler_standard.joblib')

# data :
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

sql_query = "SELECT *, CAST(alcohol_consumption AS UNSIGNED) AS alcohol_int, CAST(smoking AS UNSIGNED) AS smoking_int, CAST(has_hall AS UNSIGNED) AS hasHall_int, CAST(loves_cooking AS UNSIGNED) AS cul_skills_int FROM roommate_preferences;"
with conn.cursor() as cursor:
    cursor.execute(sql_query)
    data = cursor.fetchall()
conn.close()
data_df = pd.DataFrame(data)

data_df['alcohol_consumption'] = data_df['alcohol_int']
data_df['smoking'] = data_df['smoking_int']
data_df['has_hall'] = data_df['hasHall_int']
data_df['loves_cooking'] = data_df['cul_skills_int']

# Rename columns to match the desired names
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

# Define the columns used in the model
categorical_features = ['gender', 'rent_budget', 'alcohol', 'dist_from_uni', 'smoking', 'cul_skills', 'hasHall',
                        'maxPeople', 'numRooms']

ATTRIBUTE_WEIGHTS = {'gender': 3, "rent_budget": 1.5, "smoking": 1.5, "alcohol": 1.5}


def find_closest_roommates(input_features):
    input_df = pd.DataFrame([input_features], columns=categorical_features)
    # Scale the continuous features
    input_df['dist_from_uni'] = scaler_dist_from_uni.transform(input_df[['dist_from_uni']])
    input_df['rent_budget'] = scaler_rent_budget.transform(input_df[['rent_budget']])
    input_df[['maxPeople', 'numRooms']] = scaler_standard.transform(input_df[['maxPeople', 'numRooms']])

    # Apply weights to selected attributes
    attribute_weights = {'gender': 3, "rent_budget": 1.5, "smoking": 1.5, "alcohol": 1.5}
    for attribute, weight in attribute_weights.items():
        input_df[attribute] = input_df[attribute] * weight

    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_df)
    closest_roommates = data_df.iloc[indices[0]]

    return closest_roommates


@app.route('/find_closest_roommates', methods=['POST'])
def find_closest_roommates_api():
    try:
        input_data = request.json
        closest_roommates = find_closest_roommates(input_data)

        print(closest_roommates)
        # Remove scaler objects from the response
        closest_roommates_json = closest_roommates.to_json(orient='records')

        # Return JSON response
        return jsonify({"closest_roommates": json.loads(closest_roommates_json)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)