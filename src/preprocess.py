import configparser
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import traceback

from logger import Logger

TEST_SIZE = 0.2
SHOW_LOG = True


class DataMaker:

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "diabetes.csv")
        self.X_path = os.path.join(self.project_path, "diabetes_X.csv")
        self.y_path = os.path.join(self.project_path, "diabetes_y.csv")
        self.train_path = [os.path.join(self.project_path, "X_train.csv"), os.path.join(
            self.project_path, "y_train.csv")]
        self.test_path = [os.path.join(self.project_path, "X_test.csv"), os.path.join(
            self.project_path, "y_test.csv")]
        self.log.info("DataMaker is ready")

    def get_data(self) -> bool:
        # Reading data file, extracting features and target
        df = pd.read_csv(self.data_path)
        X, y = df.drop('Outcome', axis=1), df['Outcome']
        # Saving features and target
        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)

        # Checking whether files were successfully saved, adding names to config
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            self.config["DATA"] = {'X_data': self.X_path,
                                   'y_data': self.y_path}
            return os.path.isfile(self.X_path) and os.path.isfile(self.y_path)
        else:
            self.log.error("X and y data is not ready")
            return False

    def split_data(self, test_size=TEST_SIZE) -> bool:
        self.get_data()
        # Trying reading files, exiting on failure
        try:
            X = pd.read_csv(self.X_path, index_col=0)
            y = pd.read_csv(self.y_path, index_col=0)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        # Splitting data into train and test, saving
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0)
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config["SPLIT_DATA"] = {'X_train': self.train_path[0],
                                     'y_train': self.train_path[1],
                                     'X_test': self.test_path[0],
                                     'y_test': self.test_path[1]}
        self.log.info("Train and test data is ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.train_path[0]) and\
            os.path.isfile(self.train_path[1]) and\
            os.path.isfile(self.test_path[0]) and \
            os.path.isfile(self.test_path[1])

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        """
        Helper method, drops index and saves given dataframe
        """
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
