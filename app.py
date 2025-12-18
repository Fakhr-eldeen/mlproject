from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
from waitress import serve
from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import PredictPipeline, CustomData
application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Create CustomData object
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score', 0)),
                writing_score=float(request.form.get('writing_score', 0))
            )
            
            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("="*70)
            print("DataFrame to predict:")
            print(pred_df)
            print("="*70)
            
            # Make prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            # Print detailed error information
            print("="*70)
            print("ERROR IN PREDICTION:")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {str(e)}")
            print("="*70)
            
            import traceback
            traceback.print_exc()
            
            return render_template('home.html', results=f"Error: {str(e)}")
    
if __name__ == "__main__":
    if os.environ.get('FLASK_ENV') == 'production':
        # Use Waitress for production
        
        serve(app, host='0.0.0.0', port=8000)
    else:
        # Use Flask dev server for development
        app.run(host='0.0.0.0', port=8000, debug=True)