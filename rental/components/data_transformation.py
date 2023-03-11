from rental.entity import config_entity,artifact_entity
from rental.exception import BikeException
from rental.logger import logging
import os,sys
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from rental.config import Target_column
import pandas as pd
import numpy as np
from rental import utils



class DataTransformation:
    def __init__(self,data_transformation_config,data_ingestion_artifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise BikeException(e,sys)
            


    @classmethod
    def get_data_transformation_object(cls):
        try:
            standard_scaler = StandardScaler()
            step = [("scaler",standard_scaler)]
            pipeline = Pipeline(steps = step)
            return pipeline
        except Exception as e:
            raise BikeException(e,sys)

    def initiate_data_transformation(self):
        try:
            ### reading traing and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            ##selecting input features for train and test dataset
            train_df_input_feature = train_df.drop(Target_column,axis =1)
            
            test_df_input_feature = test_df.drop(Target_column,axis = 1)
            
            ### selecting target feature for train and test data
            train_df_target_feature = train_df[Target_column]
            logging.info(Target_column)
            test_df_target_feature = test_df[Target_column]
            


            transformation_pipeline = DataTransformation.get_data_transformation_object()
            transformation_pipeline.fit(train_df_input_feature)
            ### transformation of the input features
            input_feature_train_arr = transformation_pipeline.transform(train_df_input_feature)
            input_feature_test_arr = transformation_pipeline.transform(test_df_input_feature)


            ### transform the target feature
            target_feature_train_arr = np.array(train_df_target_feature)
            target_feature_test_arr = np.array(test_df_target_feature)


            #target encoder
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr ]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)


            utils.save_object(file_path=self.data_transformation_config.transformed_object_path,
             obj= transformation_pipeline)

            # utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            # obj=label_encoder)




            
          
            
            
            data_transformation_artifact = artifact_entity.datatransformation_artifact(
                transform_object_path=self.data_transformation_config.transformed_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
            

            )

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise BikeException(e, sys)


