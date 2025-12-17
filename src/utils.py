import os 
import sys
import pandas as pd
import numpy as np
from src.exceptions import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
def load_object(file_path):
    try:
       
        return dill.load(file_path)
    except Exception as e:
        raise e
    
def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}
        for model_name in models:
            model = models[model_name]
            
            # Get hyperparameters for this model
            para = params.get(model_name, {})
            
            # Special handling for CatBoost - skip GridSearchCV
            if "CatBoost" in model_name or "CatBoosting" in model_name:
                # Train CatBoost with default or custom params directly
                if para:
                    model.set_params(**para)
                model.fit(X_train, y_train, verbose=False)
            
            elif para:
                # For other models, use GridSearchCV
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    n_jobs=-1,
                    verbose=0
                )
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)
            else:
                # No hyperparameters, train directly
                model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)