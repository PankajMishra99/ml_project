import numpy as np   
import pandas as pd 
import os  

from src.ml_project.logger import logging 
from src.ml_project.exception import CustomException 
import sys  
from sklearn.model_selection import train_test_split 
from dataclasses import dataclass 
# print(os.getcwd())
@dataclass 
class DataIngestionConfig:
    train_data_path:str = os.path.join(os.getcwd(),'artifacts','train_data.csv')
    test_data_path:str = os.path.join(os.getcwd(),'artifacts','test_data.csv')
    raw_data_path:str = os.path.join(os.getcwd(),'artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def data_ingestion(self):
        try: 
            df=pd.read_csv(os.path.join('notebooks','raw_data.csv'))
            logging.info('Raw data reading successfully..')
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion completed successfully..') 

            return self.ingestion_config.train_data_path,self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

def main():
    ingestion=DataIngestion()
    return ingestion.data_ingestion()

if __name__=='__main__':
    main()
