import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor:

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        # cmd test options
        self.parser = argparse.ArgumentParser(description="Model")
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])

        # Reading train and test features and targets
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()

        # Loading saved models
        try:
            with open(self.config["SCALER"]["PATH"], "rb") as scaler_f:
                scaler = pickle.load(scaler_f)
            with open(self.config["LOG_REG"]["path"], "rb") as model_f:
                classifier = pickle.load(model_f)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if args.tests == "smoke":
            # Simple tests
            try:
                self.X_test = scaler.transform(self.X_test)
                score = classifier.score(self.X_test, self.y_test)
                print(f'{args.model} has {score} score')
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(
                f'{self.config["LOG_REG"]["path"]} passed smoke tests')
        elif args.tests == "func":
            # Path to testcases in json format
            tests_path = os.path.join(os.getcwd(), "tests")
            # Path to current experiment logs
            exp_path = os.path.join(os.getcwd(), "experiments")

            # Running tests and printing scores
            for test in os.listdir(tests_path):
                try:
                    data = pd.read_json(os.path.join(tests_path, test))
                    X = scaler.transform(data.drop("Outcome", axis=1))
                    y = data["Outcome"]
                    score = classifier.score(X, y)
                    print(f'Model has {score} score')
                except Exception:
                    self.log.error(traceback.format_exc())
                    sys.exit(1)
                self.log.info(
                    f'{self.config["LOG_REG"]["path"]} passed func test {test}')

                # Creating experiment info and saving with current timestamp
                exp_data = {
                    "tests": args.tests,
                    "score": str(score),
                    "X_test path": self.config["SPLIT_DATA"]["x_test"],
                    "y_test path": self.config["SPLIT_DATA"]["y_test"],
                }
                date_time = datetime.fromtimestamp(time.time())
                str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                os.mkdir(exp_dir)
                with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                    yaml.safe_dump(exp_data, exp_f, sort_keys=False)

                # Moving saved models and logs to current experiment folder
                shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                shutil.copy(self.config["LOG_REG"]["path"], os.path.join(exp_dir, f'exp_{"LOG_REG"}.sav'))
        return True


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
