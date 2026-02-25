from setuptools import find_packages,setup
from typing import List 


def find_requirements(file_path:str)->List[str]:
    """
    function used for find the list of requirments
    """
    requirements = [] 
    with open (file_path) as file:
        requirements=file.readline()
        requirements=[req.replace('\n', '') for req in requirements]

    return requirements

setup(
    name='ml_project',
    version='0.0.1',
    author='Pankaj',
    author_email='pankajmishra817395@gmail.com',
    packages=find_packages(),
    install_requires=find_requirements('requirements.txt')
)