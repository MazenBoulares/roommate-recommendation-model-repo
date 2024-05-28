import os
import shutil
import datetime
import subprocess


def delete_old_models(model_folder, days_threshold):

    model_files = [os.path.join(model_folder, f) for f in os.listdir(model_folder) if f.endswith('.joblib')]


    threshold_date = datetime.datetime.now() - datetime.timedelta(days=days_threshold)


    for model_file in model_files:

        creation_time = os.path.getctime(model_file)

        creation_date = datetime.datetime.fromtimestamp(creation_time)


        if creation_date < threshold_date:

            os.remove(model_file)
            print(f"Deleted old model: {model_file}")


def main():
    # Delete old models
    model_folder = 'models'
    days_threshold = 0.5
    delete_old_models(model_folder, days_threshold)

    # Run scheduled_generation.py
    subprocess.run(["python", "generate-models-v2.py"])


if __name__ == "__main__":
    main()
