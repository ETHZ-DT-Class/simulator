from dataclasses import dataclass
from typing import List
import pathlib
import os

from . import logger


def enclose_in_apostrophes(s: str) -> str:
    return f"'{s}'"


@dataclass
class DisplayOptions:
    screen_enable: bool
    topic_enable: bool
    mode: str
    width: int
    height: int
    compression_format: str
    rate: float
    segmentation: bool
    info_pose: bool
    info_speed: bool
    info_steps: bool
    info_time_stamp: bool

    def __post_init__(self):
        self.info_enabled = (
            self.info_pose or self.info_speed or self.info_steps or self.info_time_stamp
        )
