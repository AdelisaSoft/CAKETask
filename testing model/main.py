import os
import pickle
import pandas as pd
import numpy as np

from sklearn.svm import SVC

def load_features(path_to_model_features: str) -> list:
    """Reading model features form the given pickle file

    Args:
        path_to_model_features (str): string that tells us where \
            the model features list is, usually artifacts folder 

    Returns:
        list: List of features for the model
    """
    # Get the current directory of the main.py script
    current_directory = os.path.dirname(os.path.abspath(__file__))

    path_to_model_features = path_to_model_features.split("/")
    # Define the relative path to the 'model.pkl' file from the 'main.py' script
    model_features_path = os.path.join(current_directory, \
        path_to_model_features[0], \
            path_to_model_features[1], \
                path_to_model_features[2])

    with open(model_features_path, 'rb') as columns_file:
        model_features = pickle.load(columns_file)
        
    return model_features 
   

def load_model(path_to_model: str) -> SVC:
    """Loading model from the given pickle file

    Args:
        path_to_model (str): string that tells us where \
            the model is, usually artifacts folder 

    Returns:
        SVC: pretrained Support Vector classifier model
    """
    try:
        # Get the current directory of the main.py script
        current_directory = os.path.dirname(os.path.abspath(__file__))
        path_to_model = path_to_model.split("/")
        
        # Define the relative path to the 'model.pkl' file from the 'main.py' script
        model_file_path = os.path.join(current_directory, \
            path_to_model[0], \
                path_to_model[1], \
                    path_to_model[2])
        
        with open(model_file_path, 'rb') as model_file:
            model = pickle.load(model_file)
            
        return model
    except Exception as e:
        print(f"Unable to finish because of {e}")

def make_a_prediction(model: SVC, model_features: list, model_input: dict) -> str:
    """Function that will make predictions based on given input

    Args:
        model (SVC): Model that gives predictions
        model_features (list): List of model features
        model_input (dict): Available model inputs

    Returns:
        str: Yes/No prediction based on the model and the **kwargs
    """
    
    try:
        # As of version 3.7 dictionary is ordered, that is \
            # keys/values are returned in the order they were added.
        model_input_dict = dict()
        for feature in model_features:
            if feature in model_input:
                model_input_dict[feature] = model_input[feature]
            else:
                model_input_dict[feature] = 0
        
        print(f"Model input dict : {model_input_dict} \n")
        
        model_input = np.array([value for key, value in model_input_dict.items()])
        return model.predict(model_input.reshape(1,-1))[0]

    except Exception as e:
        print(f"Unable to finish because of {e}")
    
if __name__ == "__main__":
    
    # Load feature names
    feature_names = load_features("../artifacts/churn_model_features.pkl")
    print(f"\n\n\nModel features are {feature_names}")
    
    # Load model
    model = load_model("../artifacts/churn_model.pkl")
    print(f"\n\n\nModel we are using is {model}")
    
    model_input = {
        'value_number_of_active_months': 24, 
        'revenue': 250000, 
        'value_days_to_purchase': 4, 
        'action_create_project':3
    }
    print(f"\n\n\nFor the given input, the output is the following:\
        {make_a_prediction(model, feature_names, model_input)}")
    