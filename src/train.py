import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class Model:

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        # Loading files from preprocess stage
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)

        # Path to experiment results
        self.project_path = os.path.join(os.getcwd(), "experiments")

        # Paths to models
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.scaler_path = os.path.join(
            self.project_path, "scaler.sav")
        self.log.info("Model is ready")

    def scale(self):
        """
        Training and applying standard scaling
        """
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        params = {'path': self.scaler_path}
        return self.save_model(sc, self.scaler_path, 'SCALER', params)

    def log_reg(self, predict=False) -> bool:
        """
        Fits logistic regression and (optionally) makes predictions and prints accuracy
        """
        classifier = LogisticRegression(solver='liblinear')
        try:
            classifier.fit(self.X_train, self.y_train['Outcome'])
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        # Making predictions if specified
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")
        params = {'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        """
        Saves model and writes path to config
        """
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    model = Model()
    model.scale()
    model.log_reg(predict=True)

