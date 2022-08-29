from datarobot_drum.custom_task_interfaces import TransformerInterface
from pathlib import Path
import pickle

class CustomTask(TransformerInterface):
    
    def fit(self, X, y, row_weights=None, **kwargs):
        self.dumpedInput = X
        
        return self
    
    def save(self, artifact_directory):

        artifact_directory_path = Path(artifact_directory)
        if artifact_directory_path.exists() and artifact_directory_path.is_dir():
            with open("{}/dumped.pkl".format(artifact_directory_path), "wb") as fp:
                pickle.dump(self.dumpedInput, fp)
        
        # Helper method to handle serializing, via pickle, the CustomTask class
        self.save_task(artifact_directory, exclude=["dumpedInput"])

        return self

    def transform(self, X, **kwargs):
        return X
