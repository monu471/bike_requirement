from rental.components.data_ingestation import DataIngesation
from rental.config import mongo_client
import pymongo
import dill
from rental.logger import logging
from rental.exception import BikeException
from rental.entity import config_entity,artifact_entity
from rental.components.data_validation import DataValidation
from rental.components.model_evaluation import ModelEvaluation
from rental.components.model_trainer import ModelTrainer
from rental.components.model_pusher import ModelPusher
from rental.components.data_transformation import DataTransformation
from rental import predictor
from rental import utils
import os,sys
import pandas as pd



if __name__ =="__main__":
    try:

        train_config  = config_entity.Trainingconfig()

    #### data ingestation
        data_ingesation_config = config_entity.dataingestionconfig(train_config=train_config)
        dataingesation = DataIngesation(data_ingesation_config=data_ingesation_config)
        data_ingestaion_artifact = dataingesation.initiate_dataingesation()

    #### data validation
        data_validation_config = config_entity.datavalidationconfig(train_config=train_config)
        datavalidation = DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact=data_ingestaion_artifact)
        data_validation_artifact = datavalidation.initiate_data_validation()


    #### data transforamtion

        data_transformation_config = config_entity.datatransformationconfig(train_config=train_config)
        datatransformation = DataTransformation(data_transformation_config=data_transformation_config,data_ingestion_artifact=data_ingestaion_artifact)
        data_transformation_artifact = datatransformation.initiate_data_transformation()

    #### model training

        model_trainer_config = config_entity.ModelTrainerConfig(train_config=train_config)
        modeltrainer  = ModelTrainer(model_trainer_config= model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = modeltrainer.initiate_model_trainer()


     ### model evaluation

        model_eval_config = config_entity.ModelEvaluationConfig(train_config=train_config)
        modelevaluation = ModelEvaluation(model_eval_config= model_eval_config,dataingestion_artifact=data_ingestaion_artifact,datatransformation_artifact= data_transformation_artifact,modeltraining_artifact= model_trainer_artifact)
        model_eval_artifact = modelevaluation.initiate_model_evaluation()

     ### model pusher 
        model_pusher_config = config_entity.ModelPusherConfig(train_config = train_config)
        modelpusher = ModelPusher(model_pusher_config= model_pusher_config,datatransformation_artifact=data_transformation_artifact,modeltrainer_artifact = model_trainer_artifact)
        model_pusher_artifact = modelpusher.initiate_model_pusher()

    except Exception as e:
        raise BikeException(e,sys)    
