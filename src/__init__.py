#!/usr/bin/env python3

import os
import sys
import yaml
import pathlib

def set_duckietown_loggers_to_debug_level():
    import logging
    logging.disable(logging.DEBUG)
    import duckietown_world
    logging.disable(logging.NOTSET)
    for zuper_logger_name in ["commons", "typing", "duckietown_world", "geometry", "aido_schemas", "nodes"]:
        logger = logging.getLogger(zuper_logger_name)
        logger.setLevel(logging.INFO)
        # logger.disabled = True
        
set_duckietown_loggers_to_debug_level()

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

sys.path.append(os.path.dirname(__file__) + "/../")

from gym.envs.registration import register

from duckietown_world.resources import list_maps2

from .custom_logger import Logger

from bs4 import BeautifulSoup
from pathlib import Path
with open(Path(__file__).parent.parent / "package.xml") as f:
    __version__ = BeautifulSoup(f.read(), "xml").find('version').text

logger_name = "simulator"

with open(pathlib.Path(__file__).parent.parent / "params/logger.yaml", "r") as stream:
    logger_config = yaml.safe_load(stream)

logger_level = logger_config["level"]

logger = Logger(name=logger_name, level=logger_level)

print()
logger.info(f"simulator version {__version__}")


def on_jetson_check():
    on_jetson_str = os.environ.get("ON_JETSON")
    if on_jetson_str is None:
        on_jetson = True
    elif on_jetson_str == "true":
        on_jetson = True
    elif on_jetson_str == "false":
        on_jetson = False

    return on_jetson


def duckiebot_name():
    # system_name = os.environ.get("VEHICLE_NAME")

    # if system_name is None:
    #     # import socket
    #     # system_name = socket.gethostname()

    #     system_name = "duckiebot"
    # else:
    #     system_name = system_name.lower().replace("-", "_").replace(" ", "_")

    # duckiebot_name = f"{system_name}_sim"

    duckiebot_name = "sim_duckiebot"

    return duckiebot_name


ON_JETSON = on_jetson_check()
SIM_DUCKIEBOT_NAME = duckiebot_name()


def reg_map_env(map_name0: str, map_file: str):
    gym_id = f"Duckietown-{map_name0}-2024"

    register(
        id=f"{gym_id}-v1",
        entry_point="simulator.src:SimulatorWrapper",
        reward_threshold=400.0,
        kwargs={"map_name": map_file},
    )


for map_name, filename in list_maps2().items():
    # Register a gym environment for each map file available
    if "regress" not in filename:
        reg_map_env(map_name, filename)
