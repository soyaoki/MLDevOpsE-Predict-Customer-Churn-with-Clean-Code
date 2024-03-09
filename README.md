# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to develop a predictive model to identify customers who are likely to churn, enabling businesses to take proactive measures to retain these customers.iml

## Files and data description
```
.
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── Guide.ipynb
├── images
│   ├── eda
│   │   ├── distribution-Churn.png
│   │   ├── distribution-Customer_Age.png
│   │   ├── distribution-Marital_Status.png
│   │   ├── distribution-Total_Trans_Ct.png
│   │   └── heatmap.png
│   └── results
│       ├── classification_report-logistic_regression.png
│       ├── classification_report-random_forest.png
│       └── feature_importance-random_forest.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── README.md
├── requirements_py3.6.txt
└── requirements_py3.8.txt
```

- **churn_library.py**: Python script containing functions for finding customers likely to churn.
  
- **churn_notebook.ipynb**: Jupyter Notebook containing the initial solution to identify customer churn without implementing engineering and software best practices.

- **churn_script_logging_and_tests.py**: Python script containing unit tests for functions in `churn_library.py`, and logging errors and INFO messages.

- **data**: Directory containing the dataset used in the project.
  - **bank_data.csv**: CSV file containing the dataset used for training and testing the predictive model.

- **Guide.ipynb**: Jupyter Notebook providing a guide or instructions for executing the project.

- **images**: Directory containing images generated during exploratory data analysis (EDA) and model evaluation.
  - **eda**: Subdirectory containing EDA-related images.
    - **distribution-Churn.png**: Histogram of churn distribution.
    - **distribution-Customer_Age.png**: Histogram of customer age distribution.
    - **distribution-Marital_Status.png**: Bar plot of marital status distribution.
    - **distribution-Total_Trans_Ct.png**: Histogram of total transaction count distribution.
    - **heatmap.png**: Heatmap showing correlation between features.
  - **results**: Subdirectory containing result-related images.
    - **classification_report-logistic_regression.png**: Classification report for logistic regression model.
    - **classification_report-random_forest.png**: Classification report for random forest model.
    - **feature_importance-random_forest.png**: Feature importance plot for random forest model.

- **logs**: Directory containing log files generated during testing and logging.
  - **churn_library.log**: Log file containing errors and INFO messages from the `churn_script_logging_and_tests.py` script.

- **models**: Directory containing trained model files.
  - **logistic_model.pkl**: Pickle file containing the trained logistic regression model.
  - **rfc_model.pkl**: Pickle file containing the trained random forest classifier model.

- **README.md**: Markdown file providing an overview of the project, instructions for running the files, and guidelines for code quality.

- **requirements_py3.6.txt**: Text file listing the Python dependencies required for running the project with Python 3.6.

- **requirements_py3.8.txt**: Text file listing the Python dependencies required for running the project with Python 3.8.


## Running Files

- To prepare environment
```
[Python 3.6]
python -m pip install -r requirements_py3.6.txt

[Python 3.8]
python -m pip install -r requirements_py3.8.txt

```

- To execute model training and test
```
ipython churn_library.py
```

- To test functions
```
ipython churn_script_logging_and_tests.py
```

- To format code using PEP8 style guide
```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

- To lint code
```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```