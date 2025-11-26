# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
import json
import pickle
import os
from typing import Any, Dict, Optional

class ConfigManager:
    """
    A class to manage project configuration and data persistence.

    This class facilitates the storage and retrieval of configuration settings,
    data processing parameters (e.g., missing columns, date columns), and 
    machine learning models. It supports JSON for configuration data and 
    pickle for object serialization.

    Attributes:
        config_path (str): Path to the JSON configuration file.
        model_path (str): Directory path where models are stored.
    """

    def __init__(self, config_path: str = "config.json", model_path: str = "models/"):
        """
        Initializes the ConfigManager with paths for config and models.

        Args:
            config_path (str): The file path for the JSON configuration.
            model_path (str): The directory path for storing pickled models.
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # Ensure model directory exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        # Initialize empty config if file doesn't exist
        if not os.path.exists(self.config_path):
            self._save_json({})
    
    def __str__(self):
        return f"ConfigManager(config_path={self.config_path}, model_path={self.model_path})"

    def __repr__(self):
        return self.__str__()
    
    def _load_json(self) -> Dict[str, Any]:
        """Helper method to load the JSON configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_json(self, data: Dict[str, Any]) -> None:
        """Helper method to save data to the JSON configuration file."""
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=4)

    def set_config(self, key: str, value: Any) -> None:
        """
        Sets a configuration value for a given key.

        Args:
            key (str): The configuration key (e.g., 'missing_columns').
            value (Any): The value to store (must be JSON serializable).
        """
        config = self._load_json()
        config[key] = value
        self._save_json(config)

    def get_config(self, key: str) -> Optional[Any]:
        """
        Retrieves a configuration value for a given key.

        Args:
            key (str): The configuration key to retrieve.

        Returns:
            Optional[Any]: The value associated with the key, or None if not found.
        """
        config = self._load_json()
        return config.get(key)

    def save_model(self, model: Any, filename: str) -> str:
        """
        Serializes and saves a machine learning model to a pickle file.

        Args:
            model (Any): The model object to save.
            filename (str): The name of the file (e.g., 'random_forest.pkl').

        Returns:
            str: The full path to the saved model file.
        """
        full_path = os.path.join(self.model_path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(model, f)
        return full_path

    def load_model(self, filename: str) -> Optional[Any]:
        """
        Loads a machine learning model from a pickle file.

        Args:
            filename (str): The name of the file to load.

        Returns:
            Optional[Any]: The loaded model object, or None if the file is not found.
        """
        full_path = os.path.join(self.model_path, filename)
        if not os.path.exists(full_path):
            return None
            
        with open(full_path, 'rb') as f:
            return pickle.load(f)

