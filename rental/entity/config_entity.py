import os,sys
from rental.logger import logging
from rental.exception import BikeException
from datetime import datetime
FILE = "bike.csv"
Train_file = 'train.csv'
Test_file = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"


class Trainingconfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e :
            raise(e,sys)

class dataingestionconfig :
    def __init__(self,train_config):

        self.database = "bike"
        self.collection = "data"
        self.test_size = 0.2
        self.dataingestation_dir = os.path.join(train_config.artifact_dir,"data_ingestion")
        self.feature_store_path = os.path.join(self.dataingestation_dir,"feature_store",FILE )
        self.train_file_path = os.path.join(self.dataingestation_dir,"dataset",Train_file)
        self.test_file_path = os.path.join(self.dataingestation_dir,"dataset",Test_file)



class datavalidationconfig :
    def __init__(self,train_config):
        self.datavalidation_dir = os.path.join(train_config.artifact_dir,"data_validation")
        self.report_file_path = os.path.join(self.datavalidation_dir,"report.yaml")
        self.missing_threshold = 0.2
        self.base_file_path = r"E:\project rental\bike_requirement\bike.csv"
        

class datatransformationconfig :
    def __init__(self,train_config):
        self.datatransformation_dir = os.path.join(train_config.artifact_dir,"data_transformation" )
        self.transformed_object_path = os.path.join(self.datatransformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
        self.transformed_train_path  = os.path.join(self.datatransformation_dir,"transformed",Train_file.replace("csv","npz"))
        self.transformed_test_path = os.path.join(self.datatransformation_dir,"transformed",Test_file.replace("csv","npz"))


class ModelTrainerConfig:

    def __init__(self,train_config):
        self.model_trainer_dir = os.path.join(train_config.artifact_dir , "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model",MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.overfitting_threshold = 0.1



class ModelEvaluationConfig:
    def __init__(self,train_config):
        self.change_threshold = 0.01


class ModelPusherConfig:

    def __init__(self,train_config):
        self.model_pusher_dir = os.path.join(train_config.artifact_dir , "model_pusher")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir,TRANSFORMER_OBJECT_FILE_NAME)

