import tomllib
import pandas as pd


def getConfig() -> dict:
    """
    Get configuration from config.toml
    """
    try:
        with open("config.toml", "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
