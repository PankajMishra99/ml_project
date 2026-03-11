import os  
import sys  
import numpy as np    
import pandas as pd  
from sklearn.linear_model import LinearRegression,Ridge,Lasso 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor 
from src.ml_project.logger import logging 
from src.ml_project.exception import CustomException 
from utiles import save_object, evaluate_model 
from dataclasses import dataclass 
from sklearn.metrics import * 
import mlflow 

@dataclass 
class Model_Trainer_Config: 
    trainer_model_file_path = os.path.join('artifacts','model.pkl') 

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config = Model_Trainer_Config() 
    
    def evalute_metrics(self,actual,pred):
        mae=mean_absolute_error(actual,pred) 
        mse = mean_squared_error(actual,pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(actual,pred)
        return mae,rmse,r2 
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('split the training and testing data..') 
            x_train = train_arr[:,:-1]
            y_train= train_arr[:-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            # Regresser Model 
            models={
        'linear model':LinearRegression(),
        'Ridge':Ridge(),
        'Lasso':Lasso(),
        'k-nearest Regresser': KNeighborsRegressor(),
        'Decsion tree' : DecisionTreeRegressor(),
        'Random forest regresser': RandomForestRegressor()
                }
            
            # Parameter 
            param = {
                'linear model':{},
                'Ridge': {
                    'solver' : ['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'] 
                        },
                'Lasso': {
                    'selection' : ['cyclic','random']
                },
                'k-nearest Regresser': {
                    'n_neighbors' : [3,5,7,9,11]
                },
                'Decsion tree':{
                    'criterion' : ['squared_error','friedman_mse','absolute_error','poisson']
                },
                'Random forest regresser':{
                    'n_estimators' : [50,100,200,500]
                }
            }

            #  evalute models
            model_report:dict = evaluate_model(x_train,x_test,y_train,y_test,models,param)

            best_model_score = max(sorted(model_report.values())) 
            #  model name 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info(f'Best model name :,{best_model_name}') 

            model_names = list(param.keys())
            actual_model = ''

            for model in model_names: 
                if best_model_name==model: 
                    actual_model =actual_model + model 
            
            best_param = param[actual_model] 
            logging.info(f" Best param for {actual_model} are {best_param}") 

            #  mlflow
            mlflow.set_registry_uri('https://dagshub.com/PankajMishra99/ml_project.mlflow')

            with mlflow.start_run():
                predicted_values = best_model.predict(x_test)

                mae,rmse,r2  = self.evalute_metrics(y_test,predicted_values)
                mlflow.log_metric('Mae ',mae)
                mlflow.log_metric('RMSE',rmse)
                mlflow.log_metric('R2_score',r2) 

                mlflow.sklearn.log_model(best_model,'Model Name') 
            
            if best_model_score<0.6:
                raise CustomException('No best model found..')

            logging.info('Best model for bith training and testing dataset..')
            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                object=best_model
            )
            y_pred = best_model.predict(x_test)
            r2_sc = r2_score(y_test,y_pred)
            return r2_sc 
        except Exception as e: 
            raise CustomException(e,sys)  
        



