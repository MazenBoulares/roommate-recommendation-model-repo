import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import pymysql
import json

app = Flask(__name__)
CORS(app)


DB_CONFIG = {
    'host': "localhost",
    'port': 3306,
    'user': "root",
    'password': "root",
    'database': "entrepriseProject",
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}


MODEL_PATHS = {
    'knn': 'models/knn_model.joblib',
    'scaler_dist_from_uni': 'models/scaler_dist_from_uni.joblib',
    'scaler_rent_budget': 'models/scaler_rent_budget.joblib',
    'scaler_standard': 'models/scaler_standard.joblib'
}


CATEGORICAL_FEATURES = ['gender', 'rent_budget', 'alcohol', 'dist_from_uni', 'smoking', 'cul_skills', 'hasHall', 'maxPeople', 'numRooms']


ATTRIBUTE_WEIGHTS = {'gender': 3, "rent_budget": 1.5, "smoking": 1.5, "alcohol": 1.5}


def load_models(model_paths):

    models = {name: joblib.load(path) for name, path in model_paths.items()}
    return models


def load_data(db_config):
    """Load data from the database."""
    conn = pymysql.connect(**db_config)
    # Define your SQL query to fetch the data
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
    return data_df


def preprocess_input(input_features, models, categorical_features, attribute_weights):
    """Preprocess input features."""
    input_df = pd.DataFrame([input_features], columns=categorical_features)
    input_df['dist_from_uni'] = models['scaler_dist_from_uni'].transform(input_df[['dist_from_uni']])
    input_df['rent_budget'] = models['scaler_rent_budget'].transform(input_df[['rent_budget']])
    input_df[['maxPeople', 'numRooms']] = models['scaler_standard'].transform(input_df[['maxPeople', 'numRooms']])
    for attribute, weight in attribute_weights.items():
        input_df[attribute] = input_df[attribute] * weight
    return input_df


def find_closest_roommates(input_features, data_df, models, categorical_features, attribute_weights):
    """Find the closest roommates."""
    input_df = preprocess_input(input_features, models, categorical_features, attribute_weights)
    distances, indices = models['knn'].kneighbors(input_df)
    closest_roommates = data_df.iloc[indices[0]]
    return closest_roommates


@app.route('/find_closest_roommates', methods=['POST'])
def find_closest_roommates_api():
    try:
        input_data = request.json
        print(input_data);

        closest_roommates = find_closest_roommates(input_data, data_df, models, CATEGORICAL_FEATURES, ATTRIBUTE_WEIGHTS)
        closest_roommates_json = closest_roommates.to_json(orient='records')
        return jsonify({"closest_roommates": json.loads(closest_roommates_json)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Load models
    models = load_models(MODEL_PATHS)
    # Load data
    data_df = load_data(DB_CONFIG)
    app.run(debug=True)
