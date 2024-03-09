'''
This script performs exploratory data analysis (EDA), feature engineering, model training, prediction,
and evaluation for a bank customer churn prediction problem.

Author : Soya Aoki

Creation Date : 9th March 2024
'''


# import libraries
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # Read CSV file of inputed path
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform exploratory data analysis (EDA) on df and save figures to images folder

    input:
            df: pandas dataframe

    output:
            None
    '''
    # Display dataframe shape
    print("----- df.shape -----")
    print(df.shape)
    print("")

    # Display count of missing values in each column
    print("----- df.isnull().sum() -----")
    print(df.isnull().sum())
    print("")

    # Display summary statistics of the dataframe
    print("----- df.describe() -----")
    print(df.describe())
    print("")

    # Convert 'Attrition_Flag' to binary churn indicator
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot histogram of churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(fname='./images/eda/distribution-Churn.png')

    # Plot histogram of customer age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/distribution-Customer_Age.png')

    # Plot bar chart of marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts(normalize=True).plot(kind='bar')
    plt.savefig(fname='./images/eda/distribution-Marital_Status.png')

    # Plot density plot of total transaction count
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(fname='./images/eda/distribution-Total_Trans_Ct.png')

    # Plot heatmap of correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    Helper function to encode categorical features as new columns representing the proportion of churn for each category.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for encoded categorical features
    '''
    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()[response]

        for val in df[category]:
            column_lst.append(column_groups.loc[val])

        df[category + '_' + response] = column_lst

    return df


def perform_feature_engineering(df, response):
    '''
    Perform feature engineering and prepare data for model training.

    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    # Encode
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    df = encoder_helper(df=df, category_lst=cat_columns, response=response)

    # Set X and y for model training
    y = df[response]
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X[keep_cols] = df[keep_cols]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produce classification report for training and testing results and store report as image in images folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Show classification reports on terminal
    print('----- random forest results -----')
    print('----- test results -----')
    print(classification_report(y_test, y_test_preds_rf))
    print('----- train results -----')
    print(classification_report(y_train, y_train_preds_rf))

    print('----- logistic regression results -----')
    print('----- test results -----')
    print(classification_report(y_test, y_test_preds_lr))
    print('----- train results -----')
    print(classification_report(y_train, y_train_preds_lr))

    # Save classification reports as images
    # for random forest
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        fname='./images/results/classification_report-random_forest.png')

    # logistic regression
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        fname='./images/results/classification_report-logistic_regression.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    Create and store the feature importances in pth.

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importance-random_forest.png')


def train_models(X_train, X_test, y_train, y_test):
    '''
    Train, store model results: images + scores, and store models.

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Random forest by grid search
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Logistic regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    # Inference by train data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Inference by test data
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    # Plot ROC
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig(fname='./images/results/ROCs.png')

    # Compute and save classification reports
    classification_report_image(y_train=y_train,
                                y_test=y_test,
                                y_train_preds_lr=y_train_preds_lr,
                                y_train_preds_rf=y_train_preds_rf,
                                y_test_preds_lr=y_test_preds_lr,
                                y_test_preds_rf=y_test_preds_rf)

    # Compute and save feature importance
    feature_importance_plot(model=cv_rfc,
                            X_data=X_test,
                            output_pth='./images/results/')


if __name__ == '__main__':
    # Import data
    df_bankdata = import_data(pth='./data/bank_data.csv')

    # Perform EDA
    perform_eda(df=df_bankdata)

    # Feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df_bankdata, response='Churn')

    # Model training, prediction, and evaluation
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)
