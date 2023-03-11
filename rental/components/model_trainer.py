from typing import Optional
import os,sys 
from sklearn.ensemble import RandomForestRegressor
from rental import utils
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from rental.entity import config_entity,artifact_entity
from rental.exception import BikeException
from rental.logger import logging


class ModelTrainer:


    def __init__(self,model_trainer_config ,data_transformation_artifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact= data_transformation_artifact

        except Exception as e:
            raise BikeException(e, sys)

    
    def train_model(self,x,y):
        try:
            rf = RandomForestRegressor()
            rf.fit(x,y)
            return rf
        except Exception as e:
            raise BaseException(e, sys)


    def initiate_model_trainer(self):
        try:
            logging.info(f"Loading train and test array.")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr.")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]


            logging.info(f"Train the model")
            model = self.train_model(x=x_train,y=y_train)
            logging.info(f"calculating score of the train data")
            train_data_score = model.score(x_train, y_train)
            logging.info(f"Calculating train data mean_squre score")
            yhat_train = model.predict(x_train)
            train_data_mean_squared_error  = mean_squared_error(y_true=y_train, y_pred=yhat_train)
            logging.info(f"Calculating train data r2 score")
            train_data_r2_score = r2_score(y_pred=yhat_train,y_true=y_train)
            logging.info(f"Calculating train data adjusted r2")
            train_data_adjusted_r2_score = 1-(1-(train_data_r2_score))*(len(y_train)-1)/(len(y_train)-x_train.shape[1]-1) 


            yhat_test= model.predict(x_test)
            logging.info(f"Calculating test data mean_squre score")
            test_data_mean_squared_error  = mean_squared_error(y_true=y_test, y_pred=yhat_test)
            logging.info(f"calculating score of the test data")
            test_data_score = model.score(x_test, y_test)
            
            logging.info(f"Calculating test data r2 score")
            test_data_r2_score = r2_score(y_pred=yhat_test,y_true=y_test)

            
            logging.info(f"Calculating test data adjusted r2 score")
            # 1-(1-(model_train_R_squre))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1) 
            test_data_adjusted_r2_score = 1-(1-(test_data_r2_score))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1) 
            
            logging.info(f"train score:{train_data_score} and tests score {test_data_score}")
            #check for overfitting or underfiiting or expected score
            logging.info(f"Checking if our model is underfitting or not")
            if test_data_score<self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                expected accuracy: {self.model_trainer_config.expected_score}: model actual score: {test_data_score}")

            logging.info(f"Checking if our model is overfiiting or not")
            diff = abs(train_data_score-test_data_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff: {diff} is more than overfitting threshold {self.model_trainer_config.overfitting_threshold}")

            #save the trained model
            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Prepare the artifact")
            model_trainer_artifact  = artifact_entity.modeltraining_artifact(model_path=self.model_trainer_config.model_path, 
            train_data_score=train_data_score,train_data_mean_squared_error= train_data_mean_squared_error,train_data_r2_score= train_data_r2_score,
            train_data_adjusted_r2_score= train_data_adjusted_r2_score,test_data_score=test_data_score,test_data_mean_squared_error= test_data_mean_squared_error,test_data_r2_score= test_data_r2_score,
            test_data_adjusted_r2_score= test_data_adjusted_r2_score )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise BikeException(e, sys)
