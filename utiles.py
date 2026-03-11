import os 
import sys  
import pandas as pd  
from sklearn.metrics import * 
from sklearn.model_selection import train_test_split 
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import pickle
from sklearn.model_selection import GridSearchCV

def read_df():
    try:
        df=pd.read_csv(os.path.join(os.getcwd(),'notebooks','raw_data.csv'))
        logging.info('DataFrame readed successgully..')
        return df  
    except Exception as e:
        CustomException(e,sys)

def save_object(file_path,object):
    try:
        dir_path=os.path.dirname(file_path) 
        os.makedirs(dir_path,exist_ok=True) 

        with open(file_path,'wb') as file:
            pickle.dump(object,file)  
    except Exception as e: 
        CustomException(e,sys) 


def evaluate_model(x_train,x_test,y_train,y_test,models,param):
    try:
        report = {} 
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs=GridSearchCV(model,para,cv=4) 
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train) # train prediction

            y_test_pred = model.predict(x_test) # test prediction

            train_model_score = r2_score(y_train,y_train_pred) # train model score

            test_model_score = r2_score(y_test,y_test_pred) # test model score 

            report[list(models.keys())[i]]=test_model_score
    
        return report 
    
    except Exception as e:
        CustomException(e,sys)



