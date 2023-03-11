from rental.logger import logging
from rental.exception import BikeException
from rental.entity import config_entity,artifact_entity
import os,sys
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from rental.config import Target_column
from rental import utils

class DataValidation:
    def __init__(self,data_validation_config,data_ingestion_artifact):
        try:
     
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestation_artifact = data_ingestion_artifact
            self.validation_error = dict()

        except Exception as e :
            raise BikeException(e,sys) 


    def drop_missing_column(self,df,report_key):

        try:
            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            logging.info(f"droping the column which has missing value more than {threshold}")
            drop_column_name = null_report[null_report>threshold].index
            logging.info(f"we are droping columns{list(drop_column_name)}")
            df.drop(columns = list(drop_column_name),inplace = True)
            self.validation_error[report_key] = list(drop_column_name)
            # check length of the columns
            if len(df.columns)==0:
                return None
            return df     
        except Exception as e:
            raise BikeException(e,sys)


    def is_column_exist(self,base_df,current_df,report_key):
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_value = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f'columns: [{base_column} is not aviable')
                    missing_value.append(base_column)    


                if len(missing_value)>0:
                    self.validation_error[report_key] = missing_value
                    return False
                return True
        except Exception as e:
            raise BikeException(e,sys)       



    def data_drift(self,base_df,current_df,report_key):
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns 
            drift_report = dict()  
            
            for base_column in base_columns :
                base_data,current_data  = base_df[base_column],current_df[base_column]
                #Null hypothesis is that both column data drawn from same distrubtion
                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype}") 
                same_distribution = ks_2samp(base_data,current_data)
                if same_distribution.pvalue>0.05:
                    #We are accepting null hypothesis
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                else:
                    drift_report[base_column]={
                        "pvalues":float(same_distribution.pvalue),
                        "same_distribution":False
                    }
                    #different distribution

            self.validation_error[report_key]=drift_report


        except Exception as e:
            raise BikeException(e,sys)          
    



    def initiate_data_validation(self):
        try:
            logging.info("reading database")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            base_df = self.drop_missing_column(df = base_df,report_key="missing_value_in_base_dataset")
            logging.info("reading train dataset")
            train_df = pd.read_csv(self.data_ingestation_artifact.train_file_path)
            logging.info("reading test dataset")
            test_df = pd.read_csv(self.data_ingestation_artifact.test_file_path)
        
            logging.info("droping nan valued from train dataset")

            train_df = self.drop_missing_column(df = train_df,report_key="missing_value_in_train_data")
        
            logging.info("droping nan valued from train dataset")

            test_df = self.drop_missing_column(df = test_df,report_key="missing_value_in_test_data")

            exclude_column  = ["cnt","dteday"]
            # base_df = utils.convert_to_float(df = base_df,exclude_column=exclude_column)
            # train_df = utils.convert_to_float(df = train_df,exclude_column=exclude_column)
            # test_df = utils.convert_to_float(df = test_df,exclude_column=exclude_column)

            logging.info(f"Is all required columns present in train df")
            train_df_column_status = self.is_column_exist(base_df = base_df,current_df=train_df,report_key="missing_values_in_train_dataset")
            test_df_column_status = self.is_column_exist(base_df = base_df,current_df=test_df,report_key="missing_values_in_test_dataset")
            if train_df_column_status:
                logging.info(f"As all column are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df,report_key ="data_drift_within_train_dataset")
            if test_df_column_status:
                logging.info(f"As all column are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df,report_key="data_drift_within_test_dataset")

            #write the report
            logging.info("Write reprt in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path,
            data=self.validation_error)

            data_validation_artifact = artifact_entity.datavalidation_artifact(report_file_path=self.data_validation_config.report_file_path,)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise BikeException(e, sys)

            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       