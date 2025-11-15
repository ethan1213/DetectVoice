"""
Configuration management using YAML files.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This code is part of a research project for audio deepfake detection and adversarial robustness.
Use responsibly and in compliance with local laws and ethical guidelines.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Load and manage YAML configuration files."""

    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Dictionary with configuration parameters

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading config from: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.debug(f"Config loaded successfully")
        return config

    @staticmethod
    def save(config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            config_path: Path to save config file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to: {config_path}")

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.debug(f"Config saved successfully")

    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configurations, with override taking precedence.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
