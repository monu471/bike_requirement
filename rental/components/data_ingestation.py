from rental.logger import logging
from rental.exception import BikeException
from rental.utils import get_collection_as_dataframe
from rental.entity import config_entity,artifact_entity
from sklearn.model_selection import train_test_split
import os,sys
import pandas as pd
import numpy as np


class DataIngesation:
    def __init__(self,data_ingesation_config):
        self.data_ingestation_config = data_ingesation_config

    def initiate_dataingesation(self):
        try:
            df = get_collection_as_dataframe(database_name=self.data_ingestation_config.database,collection_name=self.data_ingestation_config.collection)
            logging.info("we are storing our dataframe into featute store ")
            
            # we want to save our dataframe into feature store path

            feature_store_dir = os.path.dirname(self.data_ingestation_config.feature_store_path)
            # #create dir if not exist
            os.makedirs(feature_store_dir,exist_ok=True)
            logging.info('we are saving the dataframe into feature store folder')
            df.to_csv(path_or_buf = self.data_ingestation_config.feature_store_path,index = False,header= True)

            logging.info("split the data into train and test split")

            train_df,test_df = train_test_split(df,test_size=self.data_ingestation_config.test_size,random_state=42)
            # ### store this dataset into dataset directory


            dataset_dir = os.path.dirname(self.data_ingestation_config.train_file_path)
            # #create dataset dir if not aviable
            os.makedirs(dataset_dir,exist_ok=True)
            logging.info("saving the train and test data ")
            
            train_df.to_csv(path_or_buf = self.data_ingestation_config.train_file_path,index = False,header = True)

            test_df.to_csv(path_or_buf = self.data_ingestation_config.test_file_path,index = False,header = True)    
        
            # #prepare artifact

            data_ingestation_artifact = artifact_entity.dataingestion_artifact(
                feature_store_path= self.data_ingestation_config.feature_store_path,
                test_file_path= self.data_ingestation_config.test_file_path,
                train_file_path= self.data_ingestation_config.train_file_path
            )
        
            logging.info(f" data_ingestation_artifact :-{data_ingestation_artifact}")
            return data_ingestation_artifact    
            return df
        except Exception as e :
            raise BikeException(e,sys)