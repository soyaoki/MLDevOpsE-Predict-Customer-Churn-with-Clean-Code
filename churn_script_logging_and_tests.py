import os
import logging
import churn_library as cls

# Set logging configure
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    Test import_data() function
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # Check if data frame has rows and columns
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    Test perform_eda() function
    '''
    df = cls.import_data("./data/bank_data.csv")

    try:
        cls.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: An error occurred")
        raise err


def test_encoder_helper():
    '''
    Test encoder_helper() function
    '''
    df = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(df)

    try:
        df = cls.encoder_helper(
            df,
            category_lst=[
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category'],
            response='Churn')
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: An error occurred")
        raise err


def test_perform_feature_engineering():
    '''
    Test perform_feature_engineering() function
    '''
    df = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(df)
    df = cls.encoder_helper(
        df,
        category_lst=[
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'],
        response='Churn')

    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df=df, response='Churn')
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: An error occurred")
        raise err


def test_train_models():
    '''
    Test train_models() function
    '''
    df = cls.import_data("./data/bank_data.csv")
    cls.perform_eda(df)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df=df, response='Churn')

    try:
        cls.train_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: An error occurred")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
