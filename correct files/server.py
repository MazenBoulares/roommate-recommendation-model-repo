import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

knn= joblib.load('models/knn_model.joblib')
scaler_dist_from_uni= joblib.load('models/scaler_dist_from_uni.joblib')
scaler_rent_budget=joblib.load('models/scaler_rent_budget.joblib')

data = pd.read_csv('notebooks/user_data_clean.csv')

data.head()


categorical_features = ['gender', 'rent_budget', 'alcohol', 'rent_budget', 'dist_from_uni', 'smoking', 'food_pref', 'cul_skills', 'bhk_1',	'bhk_2',	'bhk_3',	'bhk_4']


filtered_data = data[categorical_features]


filtered_data.head()


def find_closest_roommates(input_features):
    input_df = pd.DataFrame([input_features], columns=categorical_features)
    input_df['dist_from_uni'] = scaler_dist_from_uni.transform(input_df[['dist_from_uni']])
    input_df['rent_budget'] = scaler_rent_budget.transform(input_df[['rent_budget']])

    distances, indices = knn.kneighbors(input_df)
    closest_roommates = data.iloc[indices[0]]

    return closest_roommates





# input_roommate = {
#     'gender': '0',
#     'dist_from_uni': 0.5,
#     'rent_budget':500,
#     'alcohol': '0',
#     'smoking': '0',
#     'food_pref': '1',
#     'cul_skills': '1',
#     'bhk_1':'1',
#     'bhk_2':'1',
#     'bhk_3':'0',
#     'bhk_4':'1',
# }
#
#
# input_roommate_df = pd.DataFrame([input_roommate])
#
# closest_roommates = find_closest_roommates(input_roommate)
#
# print(closest_roommates[['name'] + categorical_features])


@app.route('/find_closest_roommates', methods=['POST'])
def find_closest_roommates_api():
    try:
        input_data = request.json
        closest_roommates = find_closest_roommates(input_data)
        result = closest_roommates[['name'] + categorical_features].to_dict(orient='records')
        return jsonify({"closest_roommates": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from JSON
#         input_data = request.get_json()
#
#         # Preprocess input data
#         input_df = preprocess_input(input_data)
#
#         # Make prediction
#         prediction = model.predict(input_df)
#
#         # Return prediction
#         return jsonify({"prediction": prediction[0] * 100})
#
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400