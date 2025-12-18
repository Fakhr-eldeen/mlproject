import sys
import os
import pandas as pd
from src.exceptions import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print(f"Loading model from: {model_path}")
            print(f"Model exists: {os.path.exists(model_path)}")
            
            print(f"Loading preprocessor from: {preprocessor_path}")
            print(f"Preprocessor exists: {os.path.exists(preprocessor_path)}")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            print("Transforming features...")
            print(f"Input features shape: {features.shape}")
            print(f"Input features columns: {features.columns.tolist()}")
            
            data_scaled = preprocessor.transform(features)
            print(f"Scaled data shape: {data_scaled.shape}")
            
            preds = model.predict(data_scaled)
            print(f"Predictions: {preds}")
            
            return preds
            
        except Exception as e:
            # Print the ACTUAL error before wrapping it
            print("="*70)
            print(f"ACTUAL ERROR IN PREDICT PIPELINE:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("="*70)
            import traceback
            traceback.print_exc()
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],  # Changed!
                "parental level of education": [self.parental_level_of_education],  # Changed!
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],  # Changed!
                "reading score": [self.reading_score],  # Changed!
                "writing score": [self.writing_score],  # Changed!
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)