from rental.predictor import ModelResolver
from rental.entity import config_entity,artifact_entity
from rental.exception import BikeException
from rental.logger import logging
from rental.utils import load_object
import pandas  as pd
import sys,os
from rental.config import Target_column
class ModelEvaluation:

    def __init__(self,model_eval_config,dataingestion_artifact,datatransformation_artifact, modeltraining_artifact):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.dataingestion_artifact = dataingestion_artifact
            self.datatransformation_artifact= datatransformation_artifact
            self.modeltraining_artifact= modeltraining_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise BikeException(e,sys)



    def initiate_model_evaluation(self):
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder

            logging.info("if saved model folder has model the we will compare "
            "which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.modelevaluation_artifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact
        



            #Finding location of transformer model and target encoder
            logging.info("Finding location of transformer model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()

            logging.info("Previous trained objects of transformer, model and target encoder")
            #Previous trained  objects
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            

            logging.info("Currently trained model objects")
            #Currently trained model objects
            current_transformer = load_object(file_path=self.datatransformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.modeltraining_artifact.model_path)
            

            train_df = pd.read_csv(self.dataingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.dataingestion_artifact.test_file_path)
            target_df = test_df[Target_column]
            y_true = target_df
            # accuracy using previous trained model
            
            input_feature_name = list(transformer.feature_names_in_)
            input_arr =transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {y_pred[:5]}")
            x_train = transformer.transform(train_df[input_feature_name])
            y_train = train_df[Target_column]
            previous_model_score = model.score(x_train,y_train)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")
           
            # accuracy using current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr =current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true = target_df
            x_test = input_arr
            print(f"Prediction using trained model: {y_pred[:5]}")
            current_model_score = current_model.score(x_test, y_true)
            logging.info(f"Accuracy using current trained model: {current_model_score}")
            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.modelevaluation_artifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"Model eval artifact: {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise BikeException(e,sys)

