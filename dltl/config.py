"""JSON configuration loading and path normalization for experiment scripts."""

import argparse
import json
from pathlib import Path


SCHEMES = ("scheme_a", "scheme_b", "scheme_c", "scheme_d")


def _resolve_path(base_dir, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_experiment_config(config_path):
    """Load a JSON config and resolve all configured paths."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    data = config["data"]
    training = config["training"]
    paths = config["paths"]
    base_dir = path.parent

    locations = {}
    for river_name, river_locations in data["locations"].items():
        locations[river_name] = {
            location_name: _resolve_path(base_dir, location_path)
            for location_name, location_path in river_locations.items()
        }

    paths["pretrained_models"] = _resolve_path(base_dir, paths["pretrained_models"])
    for group_name in ("fine_tune_models", "fine_tune_weights"):
        missing = [scheme for scheme in SCHEMES if scheme not in paths[group_name]]
        if missing:
            raise ValueError(f"Missing {group_name} paths: {', '.join(missing)}")
        paths[group_name] = {
            scheme: _resolve_path(base_dir, configured_path)
            for scheme, configured_path in paths[group_name].items()
        }

    return {
        "data": {
            "encoding": data.get("encoding", "cp949"),
            "index_column": data.get("index_column", "Date"),
            "input_variables": data["input_variables"],
            "river_data_info": {
                river_name: {"locations": river_locations}
                for river_name, river_locations in locations.items()
            },
        },
        "training": training,
        "paths": paths,
    }


def load_experiment_config_from_cli():
    """Parse ``--config`` and return a normalized experiment config."""
    parser = argparse.ArgumentParser(description="Run a DL-TL TFT experiment")
    parser.add_argument(
        "--config",
        default="config.example.json",
        help="Path to a JSON experiment configuration file",
    )
    args = parser.parse_args()
    return load_experiment_config(args.config)
