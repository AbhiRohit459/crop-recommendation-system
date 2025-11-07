from flask import Flask, request, render_template
import numpy as np
import pickle
import traceback
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Monkey patch sklearn to handle missing monotonic_cst attribute
# This fixes compatibility issues between scikit-learn versions
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Patch DecisionTreeClassifier
    _original_dt_setstate = DecisionTreeClassifier.__setstate__ if hasattr(DecisionTreeClassifier, '__setstate__') else None
    
    def _patched_dt_setstate(self, state):
        # Handle both dict and tuple state formats
        if isinstance(state, dict):
            if 'monotonic_cst' not in state:
                state = dict(state)  # Make a copy
                state['monotonic_cst'] = None
        elif isinstance(state, tuple):
            # Convert tuple to dict, add monotonic_cst, then convert back
            state_dict = dict(state) if hasattr(state, '_asdict') else {i: v for i, v in enumerate(state)}
            if 'monotonic_cst' not in state_dict and 7 not in state_dict:  # Check common positions
                # Try to add it - this is a workaround
                pass
        
        if _original_dt_setstate:
            try:
                return _original_dt_setstate(self, state)
            except AttributeError as e:
                if 'monotonic_cst' in str(e):
                    # Force add the attribute
                    if isinstance(state, dict):
                        state['monotonic_cst'] = None
                        return _original_dt_setstate(self, state)
                    else:
                        # Can't modify tuple, set attribute directly after
                        result = _original_dt_setstate(self, state)
                        if not hasattr(self, 'monotonic_cst'):
                            setattr(self, 'monotonic_cst', None)
                        return result
                raise
        else:
            if isinstance(state, dict):
                self.__dict__.update(state)
                if 'monotonic_cst' not in self.__dict__:
                    self.__dict__['monotonic_cst'] = None
    
    DecisionTreeClassifier.__setstate__ = _patched_dt_setstate
    
    # Patch RandomForestClassifier
    _original_rf_setstate = RandomForestClassifier.__setstate__ if hasattr(RandomForestClassifier, '__setstate__') else None
    
    def _patched_rf_setstate(self, state):
        if isinstance(state, dict) and 'monotonic_cst' not in state:
            state = dict(state)
            state['monotonic_cst'] = None
        if _original_rf_setstate:
            try:
                return _original_rf_setstate(self, state)
            except AttributeError as e:
                if 'monotonic_cst' in str(e):
                    if isinstance(state, dict):
                        state['monotonic_cst'] = None
                        return _original_rf_setstate(self, state)
                raise
        else:
            if isinstance(state, dict):
                self.__dict__.update(state)
    
    RandomForestClassifier.__setstate__ = _patched_rf_setstate
except Exception as e:
    print(f"Warning: Could not patch sklearn classes: {e}")

# Load model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Post-process to fix any remaining issues
    def fix_tree_attributes(obj):
        if hasattr(obj, 'tree_') and not hasattr(obj, 'monotonic_cst'):
            setattr(obj, 'monotonic_cst', None)
        if hasattr(obj, 'estimators_'):
            for estimator in obj.estimators_:
                fix_tree_attributes(estimator)
        return obj
    
    model = fix_tree_attributes(model)
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    raise

sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))


app = Flask(__name__)

# Crop dictionary with additional information
CROP_DICT = {
    1: {"name": "Rice", "season": "Kharif", "duration": "90-150 days", "water": "High"},
    2: {"name": "Maize", "season": "Kharif", "duration": "80-100 days", "water": "Moderate"},
    3: {"name": "Jute", "season": "Kharif", "duration": "120-150 days", "water": "High"},
    4: {"name": "Cotton", "season": "Kharif", "duration": "150-180 days", "water": "Moderate"},
    5: {"name": "Coconut", "season": "Year-round", "duration": "Perennial", "water": "Moderate"},
    6: {"name": "Papaya", "season": "Year-round", "duration": "Perennial", "water": "Moderate"},
    7: {"name": "Orange", "season": "Rabi", "duration": "Perennial", "water": "Moderate"},
    8: {"name": "Apple", "season": "Rabi", "duration": "Perennial", "water": "Moderate"},
    9: {"name": "Muskmelon", "season": "Summer", "duration": "70-90 days", "water": "Moderate"},
    10: {"name": "Watermelon", "season": "Summer", "duration": "80-100 days", "water": "High"},
    11: {"name": "Grapes", "season": "Year-round", "duration": "Perennial", "water": "Moderate"},
    12: {"name": "Mango", "season": "Summer", "duration": "Perennial", "water": "Moderate"},
    13: {"name": "Banana", "season": "Year-round", "duration": "Perennial", "water": "High"},
    14: {"name": "Pomegranate", "season": "Rabi", "duration": "Perennial", "water": "Low"},
    15: {"name": "Lentil", "season": "Rabi", "duration": "90-120 days", "water": "Low"},
    16: {"name": "Blackgram", "season": "Kharif", "duration": "80-90 days", "water": "Low"},
    17: {"name": "Mungbean", "season": "Kharif", "duration": "60-90 days", "water": "Low"},
    18: {"name": "Mothbeans", "season": "Kharif", "duration": "75-90 days", "water": "Low"},
    19: {"name": "Pigeonpeas", "season": "Kharif", "duration": "120-180 days", "water": "Low"},
    20: {"name": "Kidneybeans", "season": "Rabi", "duration": "90-120 days", "water": "Moderate"},
    21: {"name": "Chickpea", "season": "Rabi", "duration": "90-120 days", "water": "Low"},
    22: {"name": "Coffee", "season": "Year-round", "duration": "Perennial", "water": "High"}
}

# Input validation ranges (approximate typical ranges)
INPUT_RANGES = {
    'Nitrogen': (0, 150),
    'Phosphorus': (0, 150),
    'Potassium': (0, 250),
    'Temperature': (0, 50),
    'Humidity': (0, 100),
    'pH': (0, 14),
    'Rainfall': (0, 300)
}

def validate_inputs(form_data):
    """Validate input values against expected ranges"""
    errors = []
    warnings_list = []
    
    for field, (min_val, max_val) in INPUT_RANGES.items():
        try:
            value = float(form_data[field])
            if value < min_val or value > max_val:
                warnings_list.append(f"{field} value ({value}) is outside typical range ({min_val}-{max_val})")
        except (ValueError, KeyError):
            errors.append(f"Invalid {field} value")
    
    return errors, warnings_list

@app.route('/')
def index():
    return render_template("index.html", input_ranges=INPUT_RANGES)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Validate inputs
        errors, warnings = validate_inputs(request.form)
        if errors:
            return render_template('index.html', 
                                 error="; ".join(errors),
                                 input_ranges=INPUT_RANGES,
                                 form_data=request.form)
        
        # Get form values and convert to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)
        
        # Get prediction and probabilities
        prediction = model.predict(sc_mx_features)
        probabilities = model.predict_proba(sc_mx_features)[0] if hasattr(model, 'predict_proba') else None

        # Convert prediction to int
        predicted_value = int(prediction[0])

        # Get top 3 recommendations if probabilities are available
        recommendations = []
        if probabilities is not None:
            # Normalize probabilities to ensure they sum to 1.0 (handle any edge cases)
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities = probabilities / prob_sum
            
            # Get top 3 predictions (probabilities are 0-indexed, matching class indices)
            top_indices = np.argsort(probabilities)[::-1][:3]
            
            # Debug: Print probabilities for troubleshooting
            if app.debug:
                print(f"Probabilities shape: {probabilities.shape}")
                print(f"Probability sum: {np.sum(probabilities)}")
                print(f"Top 3 indices: {top_indices}")
                print(f"Top 3 raw probabilities: {[probabilities[i] for i in top_indices]}")
            
            for idx in top_indices:
                # The model classes are 0-indexed, but our CROP_DICT is 1-indexed
                # So we need to map: class 0 -> crop 1, class 1 -> crop 2, etc.
                crop_num = idx + 1
                if crop_num in CROP_DICT:
                    crop_info = CROP_DICT[crop_num].copy()
                    # Get the probability value (should be between 0 and 1 after normalization)
                    prob_value = float(probabilities[idx])
                    
                    # Convert to percentage and ensure it's between 0 and 100
                    confidence = min(100.0, max(0.0, round(prob_value * 100, 2)))
                    crop_info['confidence'] = confidence
                    crop_info['rank'] = len(recommendations) + 1
                    recommendations.append(crop_info)
        else:
            # Fallback to single prediction
            if predicted_value in CROP_DICT:
                crop_info = CROP_DICT[predicted_value].copy()
                crop_info['confidence'] = 100.0
                crop_info['rank'] = 1
                recommendations.append(crop_info)
            else:
                # If prediction doesn't match, try to find closest match
                # This handles potential indexing mismatches
                return render_template('index.html', 
                                     error=f"Prediction value {predicted_value} not found in crop dictionary.",
                                     input_ranges=INPUT_RANGES,
                                     form_data=request.form)

        return render_template('index.html', 
                             recommendations=recommendations,
                             warnings=warnings if warnings else None,
                             input_ranges=INPUT_RANGES,
                             form_data=request.form)
        
    except (ValueError, KeyError) as e:
        return render_template('index.html', 
                           error="Please provide valid numeric values for all fields.",
                           input_ranges=INPUT_RANGES,
                           form_data=request.form)
    except Exception as e:
        if app.debug:
            print(f"Error occurred: {str(e)}")
            print(traceback.format_exc())
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}",
                             input_ranges=INPUT_RANGES,
                             form_data=request.form)


if __name__ == "__main__":
    app.run(debug=True)