import os 
import logging 
from pathlib import Path 

logging.basicConfig(level=logging.INFO)

project_name ='ml_project' 

list_of_files = [
    f"src/{project_name}/component/__init__.py",
    f"src/{project_name}/component/data_ingestion.py",
    f"src/{project_name}/component/data_transformation.py",
    f"src/{project_name}/component/model_trainer.py",
    f"src/{project_name}/component/model_monitering.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipelines.py",
    f"src/{project_name}/pipelines/prediction_pipelines.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utiles.py",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"

]

for filepath in list_of_files: 
    filepath =Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !='':
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory for {filedir} for filename {filename}")
    
    if (not os.path.exists(filepath)  or (os.path.getsize(filepath))==0):
        with open(filepath,'w') as file:
            pass 
            logging.info(f"crteating empty file {filename}")
    else:
        logging.info("filename {filename} has already exist..")
