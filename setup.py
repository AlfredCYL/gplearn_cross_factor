from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name='gp_cross_factor',  
    version='1.0.3',  
    author='AlfredCYL',  
    author_email='alfred.yl@outlook.com', 
    description='A tool for genetic programming based cross-factor analysis',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  
    url='https://github.com/AlfredCYL/gplearn_cross_factor',  
    packages=find_packages(), 
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'joblib',
    ],
    include_package_data=True, # exclude non python files
)
