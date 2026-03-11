import os 
import sys  
from src.ml_project.exception import CustomException   
from src.ml_project.logger import logging 
from src.ml_project.component.data_ingestion import DataIngestion,DataIngestionConfig 
from src.ml_project.component.data_transformation import DataTrasformation, DataTransformationConfig 
from src.ml_project.component.model_trainer import ModelTrainer,Model_Trainer_Config 

if __name__=='__main__':
    logging.info('Excustion started..')

    try: 
        # data ingestion
        ingestion=DataIngestion()
        train_data_path,test_data_path = ingestion.data_ingestion()

        # data transformation
        data_transformation = DataTrasformation() 
        train_arr, test_arr,_ = data_transformation.initiate_data_transformer(train_data_path, test_data_path) 

        #  model trainer 
        model_trainer = ModelTrainer()
        r2_sc = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"R2 score", r2_sc)

    
    except Exception as e:
        raise CustomException(e,sys)