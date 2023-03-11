from rental.exception import BikeException

from rental.utils import load_object
from rental.predictor import ModelResolver
from datetime import datetime
from rental.config import Target_column
import os,sys 
import pandas as pd 
import numpy as np

def instance_prediction(input_list)->str:
    try:
        # df = pd.DataFrame(data=[input_dict.values()],columns=input_dict.keys())
        model_resolver = ModelResolver(model_registry='saved_models')
        transformer = load_object(model_resolver.get_latest_transformer_path())
        # input_feature_names1 = transformer.feature_names_in_
        feature = transformer.transform([input_list])

        # catergorical_transformer = load_object(model_resolver.get_latest_categorical_encoder_path())
        # input_feature_names2 = catergorical_transformer.feature_names_in_
        # df[input_feature_names2] = catergorical_transformer.transform(df[input_feature_names2])
        model = load_object(model_resolver.get_latest_model_path())
        ypred = model.predict(feature)
        # targer_encoder = load_object(model_resolver.get_latest_target_encoder_path())    
        return ypred 
    except Exception as e:
        raise BikeException(error=e, error_detail=sys)