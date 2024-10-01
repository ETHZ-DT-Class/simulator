#!/usr/bin/env python3

from typing import Union, List, Dict, Optional
import numpy as np
import numpy.typing as npt


RosParamType = Optional[Union[str, float, int, Dict, List]]

ImageType = npt.NDArray[np.uint8]  # shape (height, width, 3)
Vector2Type = npt.NDArray[np.uint]  # shape (2,)
Vector3Type = npt.NDArray[np.float64]  # shape (3,)
QuaternionType = npt.NDArray[np.float64]  # shape (4,)
Covariance3Type = npt.NDArray[np.float64]  # shape (9,)

NumberType = Union[int, float]
