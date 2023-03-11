from dataclasses import dataclass


@dataclass
class dataingestion_artifact:
    train_file_path:str
    test_file_path :str
    feature_store_path:str


@dataclass
class datavalidation_artifact:
     report_file_path:str



@dataclass
class datatransformation_artifact:
    transform_object_path:str
    transformed_train_path:str
    transformed_test_path:str




@dataclass
class modeltraining_artifact:
    model_path:str 
    train_data_score:float 
    train_data_mean_squared_error:float 
    train_data_r2_score:float
    train_data_adjusted_r2_score:float
    test_data_score:float 
    test_data_mean_squared_error:float 
    test_data_r2_score:float
    test_data_adjusted_r2_score:float


@dataclass
class modelevaluation_artifact:
    is_model_accepted:bool
    improved_accuracy:float


@dataclass
class modelpusher_artifact:
    pusher_model_dir:str 
    saved_model_dir:str

    

