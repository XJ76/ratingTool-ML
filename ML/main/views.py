import os
import numpy as np
from django.http import JsonResponse
import joblib
import pandas as pd  # Import pandas for DataFrame manipulation
from django.views.decorators.csrf import csrf_exempt

# Input data as a dictionary
input_data_1 = {
    'Operating profit (EBITDA) margin': 0.25,
    'Net profit margin': 0.15,
    'Return on Equity': 0.2,
    'Return on Assets': 0.18,
    'Total liabilities : Assets ratio': 0.5,
    'Debt: Equity ratio': 0.3,
    'Interest Cover (times)': 5,
    'Debt coverage ratio (yrs)': 2,
    'Current ratio': 2,
    'Cash : current assets': 0.2,
    'Bank Account Turnpver/Equity': 0.8,
    'Relationship with bank (no. of years)': 5,
    'Sector_Manufacturing and Production': False,
    'Sector_Primary Industries': True,
    'Sector_Retail and Distrubution': False,
    'Sector_Services': False,
    'Quality of financials_Qualified': True,
    'Quality of financials_Unaudited': False,
    'Quality of financials_Unqualified': False,
    'Strength of shareholders_Family Business': True,
    'Strength of shareholders_Quoted': False,
    'Strength of shareholders_Unquoted': False,
    'Type_Business Banking loans': False,
    'Type_CIB Loan': True,
    'Type_CIB OD': False,
    'Type_Structured Finance': False
}

# Define the rating classes
rating_classes = {
    "A+": (float('-inf'), 0.009361215),
    "A": (0.009361215, 0.012858657),
    "A-": (0.012858657, 0.016356099),
    "AB+": (0.016356099, 0.01985354),
    "AB": (0.01985354, 0.023350982),
    "AB-": (0.023350982, 0.026848423),
    "B+": (0.026848423, 0.030345865),
    "B": (0.030345865, 0.033843307),
    "B-": (0.033843307, 0.037340748),
    "BC+": (0.037340748, 0.071400192),
    "BC": (0.071400192, 0.089134417),
    "BC-": (0.089134417, 0.098001529),
    "C+": (0.098001529, 0.118551009),
    "C": (0.118551009, 0.189321872),
    "C-": (0.189321872, 0.260092735),
    "CD+": (0.260092735, 0.342873444),
    "CD": (0.342873444, 0.378686722),
    "CD-": (0.378686722, 0.4145),
    "D": (0.4145, float('inf'))
}

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        # Retrieve input data from the dictionary
        input_data = list(input_data_1.values())
        # Reshape input data to a 2D array
        input_data = np.array([input_data]).reshape(1, -1)
        # Get the path to the directory containing views.py
        current_directory = os.path.dirname(__file__)
        # Construct the absolute path to the model file
        model_path = os.path.join(current_directory, 'model (2).joblib')
        # Load the model
        model = joblib.load(model_path)
        # Use the model to make predictions
        score = model.predict_proba(input_data)[:, 1]
        # Normalize the score
        min_score = 0
        max_score = 1
        new_min = 0.0269
        new_max = 1
        normalized_score = (score - min_score) / (max_score - min_score) * (new_max - new_min) + new_min
        # Map the normalized score to rating classes
        rating = pd.cut(normalized_score, bins=[rating_classes[r][0] for r in rating_classes.keys()] + [1], labels=rating_classes.keys())
        return JsonResponse({'rating': rating[0]})
    else:
        return JsonResponse({'error': 'Only POST requests are supported.'}, status=400)
