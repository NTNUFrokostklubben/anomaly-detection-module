from pathlib import Path
import sys
import configparser


class Config:
    """
    A singleton class for reading configuration values from a config.ini file.
    The config file is read once and stored in memory for fast access.
    The class provides a get method to retrieve values from the config file based on section and key.
    The path to the config file is determined based on whether the code is running in a frozen state
    (e.g., PyInstaller) or not, allowing for flexibility in deployment.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = configparser.ConfigParser()
            cls._config.read(cls._get_config_path())
        return cls._instance

    @staticmethod
    def _get_config_path() -> Path:
        """
        Get the path to the config file, works for both dev and prod.
        :return: the path to the config file
        """
        if getattr(sys, 'frozen', False):
            return Path(sys._MEIPASS) / 'config' / 'config.ini'
        return Path(__file__).parent.parent.parent.parent / 'config' / 'config.ini'

    def get(self, section: str, key: str) -> str:
        """
        Get the configuration value for the specified section and key.
        :param section: the section in the config file, e.g., "database"
        :param key: the key of the section in the config file, e.g., "host"
        :return:  the value of the specified section and key
        """
        return self._config[section][key]