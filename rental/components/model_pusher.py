from  rental.predictor import ModelResolver
from  rental.entity.config_entity import ModelPusherConfig
from  rental.exception import BikeException
import os,sys
from  rental.utils import load_object,save_object,load_numpy_array_data,save_numpy_array_data
from  rental.logger import logging
from rental.entity.artifact_entity import datatransformation_artifact,modeltraining_artifact,modeltraining_artifact,modelpusher_artifact
class ModelPusher:

    def __init__(self,model_pusher_config,datatransformation_artifact,modeltrainer_artifact):
        try:
            logging.info(f"{'>>'*20} Model pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.datatransformation_artifact = datatransformation_artifact
            self.modeltraining_artifact = modeltrainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise BikeException(e, sys)

    def initiate_model_pusher(self,):
        try:
            #load object
            logging.info(f"Loading transformer and model")
            transformer = load_object(file_path=self.datatransformation_artifact.transform_object_path)
            model = load_object(file_path=self.modeltraining_artifact.model_path)
            

            #model pusher dir
            logging.info(f"Saving model into model pusher directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)


            #saved model dir
            logging.info(f"Saving model in saved model dir")
            transformer_path=self.model_resolver.get_latest_save_transformer_path()
            model_path=self.model_resolver.get_latest_save_model_path()

            save_object(file_path=transformer_path, obj=transformer)
            save_object(file_path=model_path, obj=model)
            

            model_pusher_artifact = modelpusher_artifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
             saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise BikeException(e, sys)


        
