import os
import yaml


def get_config():
    path_to_config = "agent/config.yaml"
    with open(path_to_config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(f'{config["gigachat"]["token"]}', "r", encoding="utf-8") as f:
        token = f.read()
        config["gigachat"]["token"] = token.strip()
    return config