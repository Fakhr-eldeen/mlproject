from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path):
    """Read the requirements from a file and return as a list."""
    requirements = []
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace('\n', ' ') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="ml_project",
    version="0.1.0",
    author='fakhr eldeen',
    author_email="fakhrldeen12@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description="A machine learning project setup",
)
