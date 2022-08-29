import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from catboost_pipeline import catboost_pipeline
from datarobot_drum.custom_task_interfaces import RegressionEstimatorInterface

class CustomTask(RegressionEstimatorInterface):
    def fit(self,
        X: pd.DataFrame,
        y: pd.Series,
        class_order: Optional[List[str]] = None,
        row_weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:


        self.estimator = catboost_pipeline(X)
        self.estimator.fit(X, y)
        
        return self

    def save(self, artifact_directory):

        # If your estimator is not pickle-able, you can serialize it using its native method.
        # Here, we'll use joblib, to illustrate this
        joblib.dump(self.estimator, "{}/catBoostEstimator.pkl".format(artifact_directory))
        
        # Helper method to handle serializing, via pickle, the CustomTask class
        self.save_task(artifact_directory, exclude=["estimator"])

        return self

    @classmethod
    def load(cls, artifact_directory):
        
        # Helper method to load the serialized CustomTask class
        custom_task = cls.load_task(artifact_directory)
        custom_task.estimator = joblib.load("{}/catBoostEstimator.pkl".format(artifact_directory))

        return custom_task

    def predict(self, X, **kwargs):
        # Note how the regression estimator only outputs one column, so no explicit column names are needed
        return pd.DataFrame(data=self.estimator.predict(X))
    
    def predict_proba(self, X, **kwargs):
        # Note that binary estimators require two columns in the output, the positive and negative class labels
        # So we need to pass in the the class names derived from the estimator as column names OR
        # we can use the class labels from DataRobot stored in
        # kwargs['positive_class_label'] and kwargs['negative_class_label']
        return pd.DataFrame(data=self.estimator.predict_proba(X))
    