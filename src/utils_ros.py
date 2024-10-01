#!/usr/bin/env python3

import rospy
import tf
import numpy as np
import cv2

from typing import Union, List, Dict, Tuple, Optional, Callable

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion, PoseStamped
from sensor_msgs.msg import CompressedImage, Imu

from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LanePose

from .simulator import LanePosition
from .Observations import *

from .custom_types import RosParamType

from . import logger


class NoParamDefaultValue:
    pass


def ros_header_from_obs(obs_header: HeaderObservation) -> Header:
    return Header(
        seq=obs_header.seq,
        stamp=rospy.Time(obs_header.stamp),
        frame_id=obs_header.frame_id,
    )


def image_msg_from_obs(camera_obs: CameraObservation) -> CompressedImage:
    image_msg = CompressedImage()
    # use numpy and cv2
    compression_format = "jpeg"
    image_msg.header = ros_header_from_obs(camera_obs.header)
    image_msg.format = compression_format
    image_msg.data = np.array(
        cv2.imencode("." + compression_format, camera_obs.image[:, :, ::-1])[1]
    ).tobytes()

    return image_msg


def image_msg_from_cv_image(cv_image: np.ndarray) -> CompressedImage:
    image_msg = CompressedImage()
    # use numpy and cv2
    compression_format = "jpeg"
    image_msg.format = compression_format
    image_msg.data = np.array(
        cv2.imencode("." + compression_format, cv_image[:, :, ::-1])[1]
    ).tobytes()

    return image_msg


def imu_msg_from_obs(imu_obs: ImuObservation) -> Imu:
    imu_msg = Imu()
    imu_msg.header = ros_header_from_obs(imu_obs.header)
    imu_msg.angular_velocity = Vector3(*imu_obs.angular_velocity)
    imu_msg.angular_velocity_covariance = imu_obs.angular_velocity_covariance
    imu_msg.linear_acceleration = Vector3(*imu_obs.linear_acceleration)
    imu_msg.linear_acceleration_covariance = imu_obs.linear_acceleration_covariance
    imu_msg.orientation = Quaternion(*imu_obs.orientation)
    imu_msg.orientation_covariance = imu_obs.orientation_covariance

    return imu_msg


def wheel_encoder_msg_from_obs(
    wheel_encoders_obs: WheelEncoderObservation,
) -> WheelEncoderStamped:
    wheel_encoder_msg = WheelEncoderStamped()
    wheel_encoder_msg.header = ros_header_from_obs(wheel_encoders_obs.header)
    wheel_encoder_msg.data = wheel_encoders_obs.ticks
    wheel_encoder_msg.resolution = wheel_encoders_obs.resolution
    wheel_encoder_msg.type = wheel_encoders_obs.type

    return wheel_encoder_msg


def pose_msg_from_pose_obs(pose_obs: PoseObservation) -> PoseStamped:
    pose_msg = PoseStamped()
    pose_msg.header = ros_header_from_obs(pose_obs.header)
    pose_msg.pose.position = Vector3(*pose_obs.position)
    pose_msg.pose.orientation = Quaternion(*pose_obs.orientation)

    return pose_msg


def wheels_cmd_msg_from_command_obs(command: CommandObservation) -> WheelsCmdStamped:
    wheels_cmd_msg = WheelsCmdStamped()
    wheels_cmd_msg.header = ros_header_from_obs(command.header)
    wheels_cmd_msg.vel_left = command.action_left
    wheels_cmd_msg.vel_right = command.action_right

    return wheels_cmd_msg


def lane_pose_msg_from_lane_pose_obs(lane_position: LanePoseObservation) -> LanePose:
    lane_pose_msg = LanePose()
    lane_pose_msg.header = ros_header_from_obs(lane_position.header)
    lane_pose_msg.d = lane_position.d
    lane_pose_msg.phi = lane_position.phi
    lane_pose_msg.in_lane = True
    lane_pose_msg.status = lane_pose_msg.NORMAL

    return lane_pose_msg


def get_quaternion_from_yaw(yaw: float) -> np.ndarray:
    return tf.transformations.quaternion_from_euler(0, 0, yaw)


def add_leading_slash(s: str) -> str:
    return s if s[0] == "/" else "/" + s


def remove_trailing_slash(s: str) -> str:
    return s if s[-1] != "/" else s[:-1]


def get_param_fn(namespace_param: str) -> Callable:
    namespace_param = remove_trailing_slash(namespace_param)
    return lambda name, default=NoParamDefaultValue(), expected_type=None: _get_param(
        "~" + namespace_param + add_leading_slash(name), default, expected_type
    )


def _get_param(
    name: str,
    default: Union[RosParamType, NoParamDefaultValue] = NoParamDefaultValue(),
    expected_type: Optional[Union[type, Tuple[type]]] = None,
) -> RosParamType:
    if rospy.has_param(name):
        param = rospy.get_param(name)
    else:
        # TODO: remove the ValueError when development is completed
        still_in_development = False
        if still_in_development:
            txt_error = f"Parameter '{name}' is not set."
            logger.error(txt_error)
            raise ValueError(logger.red(txt_error))
        if isinstance(default, NoParamDefaultValue):
            txt_error = (
                f"Parameter '{name}' is not set and no default value was provided."
            )
            logger.error(txt_error)
            raise KeyError(logger.red(txt_error))
        logger.warning(
            f"Parameter '{name}' is not set. Using default value: '{default}'"
        )
        param = default

    if isinstance(param, (str, dict)):
        param = item_2_python_variable(param)

    if expected_type is not None:
        if not isinstance(param, expected_type):
            txt_error = f"Parameter '{name}' has an unexpected type. Expected {expected_type}, got {type(param)}"
            logger.error(txt_error)
            raise TypeError(logger.red(txt_error))

    return param


def item_2_python_variable(item: RosParamType, root: bool = True) -> RosParamType:
    if isinstance(item, str):
        if item.lower() == "None":
            item = None
        elif item.lower() == "inf":
            item = float("inf")
        elif item.lower() == "-inf":
            item = float("-inf")
    elif isinstance(item, dict):
        for k, v in item.items():
            item[k] = item_2_python_variable(v, False)
    elif root:
        raise TypeError((item, type(item)))

    return item
