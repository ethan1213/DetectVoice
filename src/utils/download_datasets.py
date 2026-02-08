"""
Automated Dataset Download and Preparation Script.
This script reads the main configuration file to determine the data source
and prepares the dataset accordingly.
"""
import os
import yaml
import gdown
import subprocess
from pathlib import Path
from tqdm import tqdm
import requests

class DatasetManager:
    """
    Manages the dataset for the project by either verifying a local path
    or downloading it from a specified Google Drive folder.
    """
    def __init__(self, config: dict):
        """
        Initializes the DatasetManager with the project configuration.

        Args:
            config (dict): The loaded configuration from config.yaml.
        """
        self.config = config
        self.raw_data_dir = Path(self.config['data']['raw_data_dir'])
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self):
        """
        Main method to prepare the dataset based on the configuration.
        """
        source_type = self.config['data_source']['source_type']
        print("=" * 60)
        print("🚀 Starting Dataset Preparation")
        print(f"ℹ️  Data source type selected: '{source_type}'")
        print("=" * 60)

        if source_type == 'local':
            self._handle_local_data()
        elif source_type == 'gdrive':
            self._handle_gdrive_data()
        else:
            print(f"❌ Error: Unknown source_type '{source_type}' in 'configs/config.yaml'.")
            print("         Please use 'local' or 'gdrive'.")

    def _handle_local_data(self):
        """
        Verifies if the local data directory exists and is not empty.
        """
        local_path = Path(self.config['data_source']['local']['path'])
        print(f"ℹ️  Checking for local data at: '{local_path}'")

        if local_path.exists() and local_path.is_dir():
            # Check if the directory is empty
            if any(local_path.iterdir()):
                print(f"✅ Success: Local data directory found and is not empty.")
                print(f"   The project will use audio files from '{local_path}'.")
            else:
                print(f"⚠️ Warning: Local data directory '{local_path}' is empty.")
                print("   Please populate it with your audio datasets.")
        else:
            print(f"❌ Error: Local data directory not found at '{local_path}'.")
            print("   Please ensure the path is correct in 'configs/config.yaml' or place your data there.")
        print("=" * 60)


    def _handle_gdrive_data(self):
        """
        Downloads data from a Google Drive folder.
        """
        gdrive_config = self.config['data_source']['gdrive']
        folder_id = gdrive_config.get('folder_id')
        
        if not folder_id or folder_id == 'GDRIVE_FOLDER_ID':
            print("❌ Error: Google Drive 'folder_id' is not set in 'configs/config.yaml'.")
            print("   Please provide the ID of your Google Drive folder.")
            return

        output_path = str(self.raw_data_dir)
        print(f"📥 Downloading data from Google Drive folder: {folder_id}")
        print(f"   Saving to: '{output_path}'")
        
        try:
            # gdown will handle downloading the folder.
            # For private folders, you may need to be authenticated with Google in your environment
            # (e.g., by running 'gdown login' in your terminal first).
            gdown.download_folder(id=folder_id, output=output_path, quiet=False, use_cookies=False)
            print("✅ Google Drive download complete.")
        except Exception as e:
            print(f"❌ An error occurred during Google Drive download: {e}")
            print("   Please ensure 'gdown' is installed (`pip install gdown`).")
            print("   For private folders, you may need to authenticate with Google in your terminal.")
        print("=" * 60)


def load_config(config_path='configs/config.yaml') -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at '{config_path}'.")
        print("   Please ensure the file exists in the 'configs' directory.")
        return None
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        return None

if __name__ == '__main__':
    config = load_config()
    if config:
        manager = DatasetManager(config)
        manager.prepare_dataset()
    else:
        print("Could not proceed without a valid configuration.")