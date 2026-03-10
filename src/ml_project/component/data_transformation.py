import os 
import numpy as np 
import sys  
from dataclasses import dataclass 
import pandas as pd  
from sklearn.metrics import * 
from src.ml_project.exception import CustomException 
from src.ml_project.logger import logging 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from utiles import save_object

@dataclass
class DataTransformationConfig:
    process_file_path = os.path.join('artifacts','preprocesser.pkl')

class DataTrasformation:
    def __init__(self):
        self.process_file_path = DataTransformationConfig() 
    
    def get_data_transformer(self): 
        """
        this function used for data transformer..
        """
        try:
            num_col = ['reading score', 'writing score']
            logging.info(f"Numerical column : {num_col}")
            cat_col = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'] 
            logging.info(f'Categorical column : {cat_col}')

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scale',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    'imputer',SimpleImputer(strategy='mode'),
                    ('one_hot_encoder',OneHotEncoder())
                ]
            )

            preprocesser =ColumnTransformer(
                [
                    ('num_pipline',num_pipeline,num_col),
                    ('cat_pipeline',cat_pipeline,cat_col)
                ]
            )
            return preprocesser 
        except Exception as e:
            CustomException(e,sys)


    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path) 

            logging.info('Reading train and test file successfull..') 

            preprocesser_obj = self.get_data_transformer() 

            target_col = 'math score'
            # for training data..
            input_feature_train_df = train_df.drop([target_col],axis=1) 
            output_feature_train_df = train_df[target_col] 

            # for testing data..
            input_feature_test_df =  test_df.drop([target_col],axis=1) 
            output_feature_test_df = test_df[target_col]
            
            #  Preprocessing data 
            input_feature_train_arr = preprocesser_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=  preprocesser_obj.transform(input_feature_test_df) 

            train_arr = np.c_[
                input_feature_train_arr,np.array(output_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(output_feature_test_df)
            ] 

            save_object(
                file_path=self.process_file_path,
                object = preprocesser_obj

            )

            return (
                train_arr,
                test_arr,
                self.process_file_path
            )

        except Exception as e:
            CustomException(e,sys) 




