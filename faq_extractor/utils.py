import json
import logging
from typing import Any, Dict, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


CONFIG_JSON_REQUIRED_FIELDS = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
]


def validate_column_names(*column_names: Optional[str]) -> None:
    """
    Validate that all provided column names are unique.

    Parameters
    ----------
    column_names : Optional[str]
        Column names to be validated. Null values are ignored.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any column name is duplicated.
    """
    non_null_column_names = filter(None, column_names)
    unique_column_names = set()
    for i, column_name in enumerate(non_null_column_names, start=1):
        unique_column_names.add(column_name)
        assert len(unique_column_names) == i, (
            f"Column name {column_name!r} is duplicated. All column "
            "names must be unique."
        )


def validate_config(config_json_path: str) -> Dict[str, Any]:
    """
    Validate that the supplied configuration JSON file contains the
    required fields needed to call GPT.

    Parameters
    ----------
    config_json_path : str
        Path to the JSON configuration file.

    Returns
    -------
    Dict[str, Any]
        Loaded configuration dictionary.

    Raises
    ------
    KeyError
        If any of the required fields are missing from the configuration file.
    """
    with open(config_json_path) as f:
        config = json.load(f)
    for required_field in CONFIG_JSON_REQUIRED_FIELDS:
        if required_field not in config:
            raise KeyError(
                f"{required_field!r} is missing from {config_json_path}. The "
                f"following fields must be present: {CONFIG_JSON_REQUIRED_FIELDS}"
            )
    return config
