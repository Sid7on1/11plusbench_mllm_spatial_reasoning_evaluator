import yaml
import pathlib
import logging
import logging.config
from typing import Dict, List, Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
from logging import Formatter

class ConfigManager:
    """
    Central configuration management for models, paths, and evaluation settings.
    """

    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the ConfigManager with a configuration file.

        Args:
            config_file (str): Path to the configuration file (default: "config.yaml").
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """
        Load the configuration from the YAML file.

        Returns:
            Dict: The loaded configuration.
        """
        try:
            with open(self.config_file, "r") as file:
                config = yaml.safe_load(file)
                return config
        except FileNotFoundError:
            logging.error(f"Configuration file '{self.config_file}' not found.")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise

    def validate_paths(self) -> None:
        """
        Validate the paths in the configuration.

        Raises:
            ValueError: If any path is invalid.
        """
        for key, value in self.config.items():
            if key.startswith("path"):
                path = Path(value)
                if not path.exists():
                    logging.error(f"Invalid path: {value}")
                    raise ValueError(f"Invalid path: {value}")

    def setup_logging(self) -> None:
        """
        Set up logging with the configuration.
        """
        logging.config.dictConfig(self.config["logging"])

    def get_model_configs(self) -> Dict:
        """
        Get the model configurations.

        Returns:
            Dict: The model configurations.
        """
        return self.config["models"]

    def export_config(self) -> None:
        """
        Export the configuration to a YAML file.
        """
        with open(self.config_file, "w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

def validate_paths(config: Dict) -> None:
    """
    Validate the paths in the configuration.

    Args:
        config (Dict): The configuration to validate.
    """
    for key, value in config.items():
        if key.startswith("path"):
            path = Path(value)
            if not path.exists():
                logging.error(f"Invalid path: {value}")
                raise ValueError(f"Invalid path: {value}")

def setup_logging(config: Dict) -> None:
    """
    Set up logging with the configuration.

    Args:
        config (Dict): The logging configuration.
    """
    logging.config.dictConfig(config["logging"])

def get_model_configs(config: Dict) -> Dict:
    """
    Get the model configurations.

    Args:
        config (Dict): The configuration to retrieve model configurations from.

    Returns:
        Dict: The model configurations.
    """
    return config["models"]

def export_config(config: Dict, config_file: str = "config.yaml") -> None:
    """
    Export the configuration to a YAML file.

    Args:
        config (Dict): The configuration to export.
        config_file (str): The path to the configuration file (default: "config.yaml").
    """
    with open(config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.validate_paths()
    config_manager.setup_logging()
    model_configs = config_manager.get_model_configs()
    print(model_configs)
    config_manager.export_config()