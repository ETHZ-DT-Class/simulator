#!/usr/bin/env python3

from enum import IntEnum
from pathlib import Path
import time
import rospy
import tf
import tf2_ros
import tf_conversions

import os

import numpy as np

from src import ON_JETSON, SIM_DUCKIEBOT_NAME

from . import logger

import pyglet

pyglet.options["debug_gl"] = False

if not ON_JETSON:
    import pyglet.window

from typing import List, Optional, Tuple

from std_msgs.msg import Header
from geometry_msgs.msg import Vector3, Quaternion, Point, PoseStamped, TransformStamped
from sensor_msgs.msg import CompressedImage, Imu
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger, TriggerResponse

from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped, LanePose


def set_duckietown_loggers_to_debug_level():
    import logging

    logging.disable(logging.DEBUG)
    import gym_duckietown

    logging.disable(logging.NOTSET)
    logger = logging.getLogger("gym-duckietown")
    logger.setLevel(logging.INFO)
    # logger.disabled = True


set_duckietown_loggers_to_debug_level()

from gym_duckietown import list_maps2

from gym_duckietown.check_hw import get_graphics_information

import geometry

from pyglet.gl import *

from .simulator_wrapper import (
    SimulatorWrapper,
    MotionModelType,
    LanePosition,
    MAX_WHEEL_ANG_VEL,
)

if ON_JETSON:
    import pyglet.window

from .Observations import (
    Observations,
    Noise,
    ImuNoise,
    WheelEncoderNoise,
    AppliedCommandNoise,
    PoseObservation,
)

from .misc import DisplayOptions

from .utils_ros import *


class KeyEvent(IntEnum):
    UP = pyglet.window.key.UP
    DOWN = pyglet.window.key.DOWN
    LEFT = pyglet.window.key.LEFT
    RIGHT = pyglet.window.key.RIGHT
    SPACE = pyglet.window.key.SPACE
    BACKSPACE = pyglet.window.key.BACKSPACE
    LSHIFT = pyglet.window.key.LSHIFT
    RSHIFT = pyglet.window.key.RSHIFT
    SLASH = pyglet.window.key.SLASH
    PAGEUP = pyglet.window.key.PAGEUP
    PAGEDOWN = pyglet.window.key.PAGEDOWN
    ESCAPE = pyglet.window.key.ESCAPE
    _0 = pyglet.window.key._0
    _1 = pyglet.window.key._1
    _2 = pyglet.window.key._2
    _3 = pyglet.window.key._3
    _4 = pyglet.window.key._4
    _5 = pyglet.window.key._5
    _6 = pyglet.window.key._6
    _7 = pyglet.window.key._7
    _8 = pyglet.window.key._8
    _9 = pyglet.window.key._9

    @staticmethod
    def get_num_events():
        return [
            KeyEvent._0,
            KeyEvent._1,
            KeyEvent._2,
            KeyEvent._3,
            KeyEvent._4,
            KeyEvent._5,
            KeyEvent._6,
            KeyEvent._7,
            KeyEvent._8,
            KeyEvent._9,
        ]


DICT_NUM_KEY_EVENT_TO_LIN_ACTION = {
    KeyEvent._0: 0.0,
    KeyEvent._1: 0.11,
    KeyEvent._2: 0.22,
    KeyEvent._3: 0.33,
    KeyEvent._4: 0.44,
    KeyEvent._5: 0.55,
    KeyEvent._6: 0.66,
    KeyEvent._7: 0.77,
    KeyEvent._8: 0.88,
    KeyEvent._9: 0.99,
}


number_type = (int, float)


class SimulatorRosBridge:
    def __init__(self):

        # wait 1 second before shutting down the node
        # to allow the 'kill_simulation' service response to be sent
        rospy.on_shutdown(lambda: rospy.sleep(1))

        self.test_mode = True

        self.count_episodes = 1

        self.action = np.array([0.0, 0.0])

        self.observations: Observations = Observations()

        self.duckiebot_name = SIM_DUCKIEBOT_NAME

        self.called_kill_simulator = False
        self.called_reset_simulator = False

        self.available_maps = SimulatorWrapper.get_set_available_maps()

        self.setup_params()
        self.setup_tf_and_mesh()
        self.setup_services()
        self.setup_publishers()
        self.setup_subscribers()

        self.setup_simulator_env()

        # Register a keyboard handler
        if self.enable_additional_keyboard_control:
            if self.display_options.screen_enable:
                self.key_events_list = KeyEvent.__members__.values()
                self.num_key_events_list = KeyEvent.get_num_events()
                self.key_handler = pyglet.window.key.KeyStateHandler()
                self.env.unwrapped.window.push_handlers(self.key_handler)
            else:
                logger.debug(
                    "Keyboard control was set to true but the screen display is disabled, "
                    " hence the keyboard control is disabled."
                )

    def prepare_to_kill_simulator(self) -> None:
        self.called_kill_simulator = True

    def prepare_to_reset_simulator(self) -> None:
        self.called_reset_simulator = True

    def kill_simulator(self, msg: str) -> None:
        logger.critical(logger.bold_red_on_black(f"Killing the simulator: {msg}"))
        rospy.signal_shutdown(f"Killing the simulator: {msg}")

    def setup_simulator_env(self) -> None:
        self.env = SimulatorWrapper(
            camera_obs_enable=self.camera_obs_enable,
            imu_obs_enable=self.imu_obs_enable,
            wheel_encoders_obs_enable=self.wheel_encoders_obs_enable,
            command_obs_enable=self.command_obs_enable,
            gt_pose_obs_enable=self.gt_pose_obs_enable,
            gt_lane_pose_obs_enable=self.gt_lane_pose_obs_enable,
            camera_obs_rate=self.camera_obs_rate,
            imu_obs_rate=self.imu_obs_rate,
            wheel_encoders_obs_rate=self.wheel_encoders_obs_rate,
            wheel_encoders_resolution=self.wheel_encoders_resolution,
            command_obs_rate=self.command_obs_rate,
            gt_pose_obs_rate=self.gt_pose_obs_rate,
            gt_lane_pose_obs_rate=self.gt_lane_pose_obs_rate,
            imu_noise=self.imu_noise,
            wheel_encoder_left_noise=self.wheel_encoder_left_noise,
            wheel_encoder_right_noise=self.wheel_encoder_right_noise,
            check_drivable_enable=self.check_drivable_enable,
            check_collisions_enable=self.check_collisions_enable,
            compute_reward_enable=self.compute_reward_enable,
            seed=self.seed,
            frame_rate=self.simulation_frame_rate,
            map_name=self.map_name,
            max_steps=self.max_steps,
            domain_rand=self.domain_rand,
            duckiebot_color=self.duckiebot_color,
            camera_width=self.camera_obs_width,
            camera_height=self.camera_obs_height,
            accept_start_angle_deg=self.accept_start_angle_deg,
            full_transparency=self.full_transparency,
            distortion=self.distortion,
            window_width=self.display_options.width,
            window_height=self.display_options.height,
            style=self.style,
            frustum_filtering=self.frustum_filtering,
            frustum_filtering_min_arccos_threshold=self.frustum_filtering_min_arccos_threshold,
            depth_filtering_factor=self.depth_filtering_factor,
            depth_texture_resize_factor=self.depth_texture_resize_factor,
            distant_texture_resize_factor=self.distant_texture_resize_factor,
            force_texture_resize_floor=self.force_texture_resize_floor,
            skip_frame_buffer_copy=self.skip_frame_buffer_copy,
            init_pose=self.init_pose,
            motion_model=self.motion_model,
            motion_model_delay=self.motion_model_delay,
            motion_model_wheel_distance=self.motion_model_wheel_distance,
            motion_model_wheel_radius_left=self.motion_model_wheel_radius_left,
            motion_model_wheel_radius_right=self.motion_model_wheel_radius_right,
            imu_is_rotated=self.imu_is_rotated,
        )

        self.rospy_rate = rospy.Rate(self.realtime_factor * self.simulation_frame_rate)

        self.print_info()

        # self.reset()

        self.render_external()

    def print_info(self) -> None:
        if not self.print_startup_log:
            print()
            logger.info(
                "'print_startup_log' is set to False, hence the startup log will not be printed."
            )
            print()
            return

        print("\n------------------------------------ \n")

        information = get_graphics_information()
        using_gpu = any(
            [
                "nvidia" in information[field].lower()
                for field in ["renderer", "version"]
            ]
        )
        renderer = information["renderer"]
        logger.info(
            f"Running on {logger.bold('Jetson Nano' if ON_JETSON else 'laptop')}"
            f" with NVIDIA GPU {logger.blink_green('ENABLED') if using_gpu else logger.blink_red('DISABLED')},"
            f" using {logger.bold(renderer)}.\n"
        )

        logger.info(
            f"Simulation updates at specified simulated "
            + (
                f"frame rate of {self.simulation_frame_rate} Hz"
                if self.simulation_update_use == "frame_rate"
                else f"delta time step of {1/self.simulation_frame_rate} seconds"
            )
            + f", using the {logger.bold_green_on_black(self.env.get_motion_model())} motion model."
        )
        logger.info(
            f"The simulator is using the map {logger.bold_green_on_black(self.map_name)} and the texture style {logger.bold_green_on_black(self.style)}.\n"
        )

        logger.info(
            f"'repeat action until new' is {logger.blink_green('ENABLED') if self.repeat_action_until_new else logger.blink_red('DISABLED')}: the wheel command "
            + (
                f"is applied indefinetely until a new command is received."
                if self.repeat_action_until_new
                else f"is applied for one time step only."
            )
        )

        logger.info(
            f"'reset the ros time when environment is reset' is "
            + (
                logger.blink_green("ENABLED")
                if self.reset_ros_time_when_env_is_reset
                else logger.blink_red("DISABLED")
            )
            + ": when the environment is reset, the ros stamp of the topics will "
            + ("" if self.reset_ros_time_when_env_is_reset else "not ")
            + "reset to 0."
        )

        logger.info(
            f"'reset the action when environment is reset' is "
            + (
                logger.blink_green("ENABLED")
                if self.reset_action_when_env_is_reset
                else logger.blink_red("DISABLED")
            )
            + ": when the environment is reset, the action will "
            + ("" if self.reset_action_when_env_is_reset else "not ")
            + "reset to [0, 0].\n"
        )

        logger.info(
            f"Reset mode is {logger.bold_green_on_black(self.reset_mode)}"
            + (
                ""
                if self.reset_mode == "kill"
                else (
                    f" with drivable area check {logger.blink_green('ENABLED') if self.check_drivable_enable else logger.blink_red('DISABLED')}"
                    f" and collisions check {logger.blink_green('ENABLED') if self.check_collisions_enable else logger.blink_red('DISABLED')}"
                )
            )
            + ".\n"
        )

        def print_sensor_info(sensor_name) -> None:
            def camel_to_snake(name: str) -> str:
                return "".join(
                    ["_" + c.lower() if c.isupper() else c for c in name]
                ).lstrip("_")

            sensor_name = camel_to_snake(sensor_name).replace("ground_truth", "gt")

            if "wheel_encoder" in sensor_name:
                sensor_obs_enable = getattr(self, f"wheel_encoders_obs_enable")
                sensor_obs_rate = getattr(self, f"wheel_encoders_obs_rate")
            else:
                sensor_obs_enable = getattr(self, f"{sensor_name}_obs_enable")
                sensor_obs_rate = getattr(self, f"{sensor_name}_obs_rate")
            sensor_noise = getattr(self, f"{sensor_name}_noise", None)
            sensor_topic = getattr(self, f"name_pub_{sensor_name}_topic")

            indent_specific_noise_info = " " * 10
            newline = "\n"

            logger.info(
                f"  - {sensor_name}: {logger.blink_green('ENABLED') if sensor_obs_enable else logger.blink_red('DISABLED')}"
                f"{f' at {sensor_obs_rate} Hz - publishing to {logger.bold_green_on_black(sensor_topic)}' if sensor_obs_enable else ''}"
            )
            if not sensor_obs_enable or sensor_noise is None:
                return
            if isinstance(sensor_noise, ImuNoise):
                txt_noise_list = (
                    str(sensor_noise)
                    .replace("ENABLED", logger.blink_green("ENABLED"))
                    .replace("DISABLED", logger.blink_red("DISABLED"))
                    .splitlines()
                )
                logger.info(f"      with noise:")
                for txt_noise in txt_noise_list:
                    logger.info(
                        f"{indent_specific_noise_info}{txt_noise}"
                        if sensor_noise.enable
                        else logger.blink_red("DISABLED")
                    )
            else:
                txt_noise = f"{sensor_noise if sensor_noise.enable else logger.blink_red('DISABLED')}"
                logger.info(f"      with noise: {txt_noise}")

        logger.info("Generating observations:")
        for sensor in Observations.__annotations__.keys():
            print_sensor_info(sensor)
        print()
        logger.info(
            f"External displaying of the simulation"
            + (
                f" with mode {logger.bold_green_on_black(self.display_options.mode)} at {self.display_options.rate} Hz"
                if (self.display_options.topic_enable)
                else ""
            )
            + " is: "
            + (
                f"{logger.blink_green('ENABLED')} - publishing to {logger.bold_green_on_black(self.name_pub_external_window_topic)}"
                if self.display_options.topic_enable
                else logger.blink_red("DISABLED")
            )
            + ".\n"
        )

    def print_info_sim_is_running(self) -> None:
        print()
        logger.info(
            f"Simulation is running at (if possible) {self.realtime_factor} real-time factor. Check published topics."
            f" Publish a message to the topic {logger.bold_green_on_black(self.name_sub_wheel_cmd_topic)}"
            f" to control your duckiebot {logger.bold(self.duckiebot_name)}.\n"
        )

        post_startup_log_warning_tag = "<warning>"
        if len(post_startup_log_warning_tag) > 0:
            if (
                self.post_startup_log_text[: len(post_startup_log_warning_tag)]
                == post_startup_log_warning_tag
            ):
                logger.warning_whole(
                    f"{self.post_startup_log_text[len(post_startup_log_warning_tag):]}"
                )
            else:
                logger.info(self.post_startup_log_text)
            print()

    def setup_params(self) -> None:

        self.read_params()
        self.check_params()
        self.finish_setup_params()

    def read_params(self) -> None:
        def add_leading_slash(s: str) -> str:
            return s if s[0] == "/" else "/" + s

        # topics params
        ns_topics_params = "topics"
        topic_prefix = "/" + self.duckiebot_name
        get_topic_param = (
            lambda param_name, expected_type=None: topic_prefix
            + add_leading_slash(
                get_param_fn(ns_topics_params)(param_name, expected_type=expected_type)
            )
        )

        self.name_sub_wheel_cmd_topic = get_topic_param(
            "/sub/wheel_cmd", expected_type=str
        )
        self.name_pub_camera_topic = get_topic_param("/pub/camera", expected_type=str)
        self.name_pub_imu_topic = get_topic_param("/pub/imu", expected_type=str)
        self.name_pub_wheel_encoder_left_topic = get_topic_param(
            "/pub/wheel_encoder_left", expected_type=str
        )
        self.name_pub_wheel_encoder_right_topic = get_topic_param(
            "/pub/wheel_encoder_right", expected_type=str
        )
        self.name_pub_external_window_topic = get_topic_param(
            "/pub/simulation_external_window", expected_type=str
        )
        self.name_pub_command_topic = get_topic_param("/pub/command", expected_type=str)
        self.name_pub_gt_pose_topic = get_topic_param("/pub/gt_pose", expected_type=str)
        self.name_pub_gt_lane_pose_topic = get_topic_param(
            "/pub/lane_pose", expected_type=str
        )
        service_prefix = "/simulator"
        get_topic_service_param = (
            lambda param_name, expected_type=None: service_prefix
            + add_leading_slash(
                get_param_fn(ns_topics_params)(param_name, expected_type=expected_type)
            )
        )
        self.name_srv_server_reset_sim = get_topic_service_param(
            "/srv/reset_sim", expected_type=str
        )
        self.name_srv_server_kill_sim = get_topic_service_param(
            "/srv/kill_sim", expected_type=str
        )

        # simulator params
        ns_simulator_params = "simulator"
        get_sim_param = get_param_fn(ns_simulator_params)
        seed = get_sim_param("/seed", 0, int)
        if seed == 0:
            logger.warning(
                "Seed parameter is set to 0, hence the seed will be set to None."
            )
            self.seed = None
        else:
            self.seed = seed
        self.map_name = get_sim_param("/map_name", "loop_empty", expected_type=str)
        self.max_steps = get_sim_param(
            "/max_steps", float("inf"), expected_type=number_type
        )
        self.domain_rand = get_sim_param("/domain_rand", 0, expected_type=bool)
        self.accept_start_angle_deg = get_sim_param(
            "/accept_start_angle_deg", 4, expected_type=number_type
        )
        self.full_transparency = get_sim_param(
            "/full_transparency", True, expected_type=bool
        )
        self.distortion = get_sim_param("/distortion", False, expected_type=bool)
        self.enable_additional_keyboard_control = get_sim_param(
            "/additional_keyboard_control", False, expected_type=bool
        )
        self.duckiebot_color = get_sim_param(
            "/duckiebot_color", "red", expected_type=str
        )
        init_pose_enable = get_sim_param("/init_pose/enable", False, expected_type=bool)
        if init_pose_enable:
            init_pose_x = get_sim_param("/init_pose/x", 0, expected_type=number_type)
            init_pose_y = get_sim_param("/init_pose/y", 0, expected_type=number_type)
            init_pose_theta = get_sim_param(
                "/init_pose/theta", 0, expected_type=number_type
            )
            self.init_pose = np.array([init_pose_x, init_pose_y, init_pose_theta])
        else:
            self.init_pose = None
        self.check_drivable_enable = get_sim_param(
            "/check/drivable", True, expected_type=bool
        )
        self.check_collisions_enable = get_sim_param(
            "/check/collisions", True, expected_type=bool
        )
        self.compute_reward_enable = get_sim_param(
            "/compute_reward", False, expected_type=bool
        )
        self.motion_model = get_sim_param(
            "/motion_model/type", "dynamics", expected_type=str
        )
        self.motion_model_delay = get_sim_param(
            "/motion_model/params/delay", 0, expected_type=number_type
        )
        self.motion_model_wheel_distance = get_sim_param(
            "/motion_model/params/wheel_distance", 0.102, expected_type=number_type
        )
        self.motion_model_wheel_radius_left = get_sim_param(
            "/motion_model/params/wheel_radius_left", 0.0335, expected_type=number_type
        )
        self.motion_model_wheel_radius_right = get_sim_param(
            "/motion_model/params/wheel_radius_right", 0.0335, expected_type=number_type
        )
        self.applied_command_noise = self.get_applied_command_noise_params()
        self.applied_command_noise_min_action_threshold = get_sim_param(
            "/motion_model/applied_command_noise/min_action_threshold",
            0.1,
            number_type,
        )
        self.simulation_update_use = get_sim_param("/update/use", expected_type=str)
        if self.simulation_update_use == "frame_rate":
            self.simulation_frame_rate = get_sim_param(
                "/update/frame_rate", 20, expected_type=number_type
            )
        else:
            self.simulation_frame_rate = 1 / get_sim_param(
                "/update/delta_time_step", 1 / 20, expected_type=number_type
            )
        self.frustum_filtering = get_sim_param(
            "/rendering/frustum_filtering", False, expected_type=bool
        )
        self.frustum_filtering_min_arccos_threshold = get_sim_param(
            "/rendering/frustum_filtering_min_arccos_threshold",
            0,
            expected_type=number_type,
        )
        self.depth_filtering_factor = get_sim_param(
            "/rendering/depth_filtering_factor", 5, expected_type=number_type
        )
        self.depth_texture_resize_factor = get_sim_param(
            "/rendering/depth_texture_resize_factor", 4, expected_type=number_type
        )
        self.distant_texture_resize_factor = get_sim_param(
            "/rendering/distant_texture_resize_factor", 1 / 4, expected_type=number_type
        )
        self.force_texture_resize_floor = get_sim_param(
            "/rendering/force_texture_resize_floor", True, expected_type=bool
        )
        self.skip_frame_buffer_copy = get_sim_param(
            "/rendering/skip_frame_buffer_copy", True, expected_type=bool
        )
        self.style = get_sim_param(
            "/rendering/texture_style", "photos", expected_type=str
        )
        screen_enable = (
            get_sim_param("/display/screen_enable", False, expected_type=bool)
            and not ON_JETSON
        )
        topic_enable = get_sim_param("/display/topic_enable", False, expected_type=bool)
        display_mode = get_sim_param("/display/mode", "human", expected_type=str)
        display_rate = get_sim_param("/display/rate", 10, expected_type=number_type)
        display_enable_segmentation = get_sim_param(
            "/display/enable_segmentation", False, expected_type=bool
        )
        self.display_reuse_camera_obs_if_possible = get_sim_param(
            "/display/reuse_camera_obs_if_possible", True, expected_type=bool
        )
        display_width = get_sim_param("/display/width", 320, expected_type=int)
        display_height = get_sim_param("/display/height", 240, expected_type=int)
        compression_format = get_sim_param(
            "/display/compression_format", "jpeg", expected_type=str
        )
        display_info_pose = get_sim_param(
            "/display/show/pose", False, expected_type=bool
        )
        display_info_speed = get_sim_param(
            "/display/show/speed", False, expected_type=bool
        )
        display_info_steps = get_sim_param(
            "/display/show/steps", False, expected_type=bool
        )
        display_info_time_stamp = get_sim_param(
            "/display/show/time_stamp", False, expected_type=bool
        )
        self.display_options = DisplayOptions(
            screen_enable=screen_enable,
            topic_enable=topic_enable,
            mode=display_mode,
            width=display_width,
            height=display_height,
            compression_format=compression_format,
            rate=display_rate,
            segmentation=display_enable_segmentation,
            info_pose=display_info_pose,
            info_speed=display_info_speed,
            info_steps=display_info_steps,
            info_time_stamp=display_info_time_stamp,
        )
        ns_simulator_observations_params = ns_simulator_params + "/observations"
        get_sim_obs_param = get_param_fn(ns_simulator_observations_params)
        self.camera_obs_enable = get_sim_obs_param(
            "/camera/enable", True, expected_type=bool
        )
        self.camera_obs_rate = get_sim_obs_param(
            "/camera/rate", 20, expected_type=number_type
        )
        self.camera_obs_width = get_sim_obs_param(
            "/camera/width", 640, expected_type=int
        )
        self.camera_obs_height = get_sim_obs_param(
            "/camera/height", 480, expected_type=int
        )
        self.camera_obs_compression_format = get_sim_obs_param(
            "/camera/compression_format", "jpeg", expected_type=str
        )
        self.imu_obs_enable = get_sim_obs_param("/imu/enable", True, expected_type=bool)
        self.imu_obs_rate = get_sim_obs_param(
            "/imu/rate", 100, expected_type=number_type
        )
        self.imu_is_rotated = get_sim_obs_param(
            "/imu/is_rotated", False, expected_type=bool
        )
        self.wheel_encoders_obs_enable = get_sim_obs_param(
            "/wheel_encoders/enable", True, expected_type=bool
        )
        self.wheel_encoders_obs_rate = get_sim_obs_param(
            "/wheel_encoders/rate", 50, expected_type=number_type
        )
        self.wheel_encoders_resolution = get_sim_obs_param(
            "/wheel_encoders/ticks_per_revolution", 32, expected_type=int
        )
        self.gt_pose_obs_enable = get_sim_obs_param(
            "/gt_pose/enable", False, expected_type=bool
        )
        self.gt_pose_obs_rate = get_sim_obs_param(
            "/gt_pose/rate", 20, expected_type=number_type
        )
        self.gt_lane_pose_obs_enable = get_sim_obs_param(
            "/lane_pose/enable", False, expected_type=bool
        )
        self.gt_lane_pose_obs_rate = get_sim_obs_param(
            "/lane_pose/rate", 20, expected_type=number_type
        )
        self.command_obs_enable = get_sim_obs_param(
            "/command/enable", False, expected_type=bool
        )
        self.command_obs_rate = get_sim_obs_param(
            "/command/rate", 20, expected_type=number_type
        )
        self.imu_noise = self.get_imu_noise_params()
        self.wheel_encoder_left_noise, self.wheel_encoder_right_noise = (
            self.get_wheel_encoders_noise_params()
        )

        # bridge params
        ns_bridge_params = "bridge"
        get_bridge_param = get_param_fn(ns_bridge_params)
        self.realtime_factor = get_bridge_param(
            "/realtime_factor", 1.0, expected_type=number_type
        )
        self.repeat_action_until_new = get_bridge_param(
            "/repeat_action_until_new", True, expected_type=bool
        )
        self.reset_mode = get_bridge_param("/reset/mode", "reset", expected_type=str)
        self.reset_ros_time_when_env_is_reset = get_bridge_param(
            "/reset/reset_ros_time_when_env_is_reset", True, expected_type=bool
        )
        self.reset_action_when_env_is_reset = get_bridge_param(
            "/reset/reset_action_when_env_is_reset", True, expected_type=bool
        )
        self.path_folder_mesh_resources = get_bridge_param(
            "/meshes_folder_path", expected_type=str
        )
        self.path_duckiebot_mesh_resource = (
            self.path_folder_mesh_resources
            + f"/duckiebot/duckiebot-{self.duckiebot_color}.dae"
        )
        self.print_startup_log = get_bridge_param(
            "/print_startup_log", True, expected_type=bool
        )
        self.post_startup_log_text = get_bridge_param(
            "/post_startup_log_text", "", expected_type=str
        )

    def get_imu_noise_params(self) -> ImuNoise:
        ns_imu_noise_params = "simulator/observations/noise/imu"
        get_imu_noise_param = get_param_fn(ns_imu_noise_params)
        enable_imu_obs_noise = get_imu_noise_param("/enable", False, expected_type=bool)
        imu_obs_lin_acc_noise_white_noise_sigma = get_imu_noise_param(
            "/lin_acc/white_noise/sigma", 0, expected_type=number_type
        )
        imu_obs_lin_acc_noise_bias_mu = get_imu_noise_param(
            "/lin_acc/bias/mu", 0, expected_type=number_type
        )
        imu_obs_lin_acc_noise_bias_sigma = get_imu_noise_param(
            "/lin_acc/bias/sigma", 0, expected_type=number_type
        )
        imu_obs_lin_acc_noise = Noise(
            enable=enable_imu_obs_noise,
            white_noise_sigma=imu_obs_lin_acc_noise_white_noise_sigma,
            bias_mu=imu_obs_lin_acc_noise_bias_mu,
            bias_sigma=imu_obs_lin_acc_noise_bias_sigma,
        )

        imu_obs_ang_vel_noise_white_noise_sigma = get_imu_noise_param(
            "/ang_vel/white_noise/sigma", 0, expected_type=number_type
        )
        imu_obs_ang_vel_noise_bias_mu = get_imu_noise_param(
            "/ang_vel/bias/mu", 0, expected_type=number_type
        )
        imu_obs_ang_vel_noise_bias_sigma = get_imu_noise_param(
            "/ang_vel/bias/sigma", 0, expected_type=number_type
        )
        imu_obs_ang_vel_noise = Noise(
            enable=enable_imu_obs_noise,
            white_noise_sigma=imu_obs_ang_vel_noise_white_noise_sigma,
            bias_mu=imu_obs_ang_vel_noise_bias_mu,
            bias_sigma=imu_obs_ang_vel_noise_bias_sigma,
        )

        imu_obs_orientation_noise_white_noise_sigma = get_imu_noise_param(
            "/orientation/white_noise/sigma", 0, expected_type=number_type
        )
        imu_obs_orientation_noise_bias_mu = get_imu_noise_param(
            "/orientation/bias/mu", 0, expected_type=number_type
        )
        imu_obs_orientation_noise_bias_sigma = get_imu_noise_param(
            "/orientation/bias/sigma", 0, expected_type=number_type
        )
        imu_obs_orientation_noise = Noise(
            enable=enable_imu_obs_noise,
            white_noise_sigma=imu_obs_orientation_noise_white_noise_sigma,
            bias_mu=imu_obs_orientation_noise_bias_mu,
            bias_sigma=imu_obs_orientation_noise_bias_sigma,
        )

        return ImuNoise(
            lin_acc=imu_obs_lin_acc_noise,
            ang_vel=imu_obs_ang_vel_noise,
            orientation=imu_obs_orientation_noise,
        )

    def get_wheel_encoders_noise_params(self) -> List[WheelEncoderNoise]:
        ns_wheel_encoders_noise_params = "simulator/observations/noise/wheel_encoders"
        get_wheel_encoders_noise_param = get_param_fn(ns_wheel_encoders_noise_params)
        enable_wheel_encoders_obs_noise = get_wheel_encoders_noise_param(
            "/enable", False, expected_type=bool
        )
        wheel_encoder_left_obs_noise_white_noise_sigma = get_wheel_encoders_noise_param(
            "/left/white_noise/sigma", 0, expected_type=number_type
        )
        wheel_encoder_left_obs_noise_bias_mu = get_wheel_encoders_noise_param(
            "/left/bias/mu", 0, expected_type=number_type
        )
        wheel_encoder_left_obs_noise_bias_sigma = get_wheel_encoders_noise_param(
            "/left/bias/sigma", 0, expected_type=number_type
        )

        wheel_encoder_right_obs_noise_white_noise_sigma = (
            get_wheel_encoders_noise_param(
                "/right/white_noise/sigma", 0, expected_type=number_type
            )
        )
        wheel_encoder_right_obs_noise_bias_mu = get_wheel_encoders_noise_param(
            "/right/bias/mu", 0, expected_type=number_type
        )
        wheel_encoder_right_obs_noise_bias_sigma = get_wheel_encoders_noise_param(
            "/right/bias/sigma", 0, expected_type=number_type
        )

        wheel_encoder_left_noise = WheelEncoderNoise(
            enable=enable_wheel_encoders_obs_noise,
            white_noise_sigma=wheel_encoder_left_obs_noise_white_noise_sigma,
            bias_mu=wheel_encoder_left_obs_noise_bias_mu,
            bias_sigma=wheel_encoder_left_obs_noise_bias_sigma,
        )
        wheel_encoder_right_noise = WheelEncoderNoise(
            enable=enable_wheel_encoders_obs_noise,
            white_noise_sigma=wheel_encoder_right_obs_noise_white_noise_sigma,
            bias_mu=wheel_encoder_right_obs_noise_bias_mu,
            bias_sigma=wheel_encoder_right_obs_noise_bias_sigma,
        )

        return [wheel_encoder_left_noise, wheel_encoder_right_noise]

    def get_applied_command_noise_params(self) -> AppliedCommandNoise:
        ns_applied_command_noise_params = "simulator/motion_model/applied_command_noise"
        get_applied_command_noise_param = get_param_fn(ns_applied_command_noise_params)
        enable_applied_command_obs_noise = get_applied_command_noise_param(
            "/enable", False, expected_type=bool
        )
        applied_command_obs_noise_white_noise_sigma = get_applied_command_noise_param(
            "/white_noise/sigma", 0, expected_type=number_type
        )
        applied_command_obs_noise_bias_mu = get_applied_command_noise_param(
            "/bias/mu", 0, expected_type=number_type
        )
        applied_command_obs_noise_bias_sigma = get_applied_command_noise_param(
            "/bias/sigma", 0, expected_type=number_type
        )

        return AppliedCommandNoise(
            enable=enable_applied_command_obs_noise,
            white_noise_sigma=applied_command_obs_noise_white_noise_sigma,
            bias_mu=applied_command_obs_noise_bias_mu,
            bias_sigma=applied_command_obs_noise_bias_sigma,
        )

    def check_params(self) -> None:

        list_error_msgs = []
        list_error_param_names = []

        try:

            # Check misc parameters
            if self.simulation_frame_rate <= 0:
                param_name = "Simulation frame rate"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"{param_name} {self.simulation_frame_rate} must be a positive number."
                )
            if self.motion_model_delay < 0:
                param_name = "Motion model delay"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"M{param_name} {self.motion_model_delay} must be a non-negative number."
                )
            if self.motion_model_wheel_distance <= 0:
                param_name = "Motion model wheel distance"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"{param_name} {self.motion_model_wheel_distance} must be a positive number."
                )
            if self.motion_model_wheel_radius_left <= 0:
                param_name = "Motion model wheel radius left"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"{param_name} {self.motion_model_wheel_radius_left} must be a positive number."
                )
            if self.motion_model_wheel_radius_right <= 0:
                param_name = "Motion model wheel radius right"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"{param_name} {self.motion_model_wheel_radius_right} must be a positive number."
                )
            if self.realtime_factor <= 0:
                param_name = "Realtime factor"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    f"{param_name} {self.realtime_factor} must be a positive number."
                )
            if self.camera_obs_enable:
                if not self.camera_obs_width > 0:
                    param_name = "Camera width"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.camera_obs_width} must be a positive integer."
                    )
                if not self.camera_obs_height > 0:
                    param_name = "Camera height"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.camera_obs_height} must be a positive integer."
                    )

            if self.wheel_encoders_obs_enable:
                if not self.wheel_encoders_resolution > 0:
                    param_name = "Wheel encoders resolution"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"param_name {self.wheel_encoders_resolution} must be a positive integer."
                    )

            if self.depth_texture_resize_factor >= 0:
                if self.distant_texture_resize_factor <= 0:
                    param_name = "Distant texture resize factor"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.distant_texture_resize_factor} must be a positive number."
                    )
                elif not (
                    self.distant_texture_resize_factor % 2 == 0
                    or np.isclose((1 / self.distant_texture_resize_factor) % 2, 0)
                ):
                    param_name = "Distant texture resize factor"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.distant_texture_resize_factor} must be multiple of 2 or divisor of 1/2."
                    )

            if self.display_options.screen_enable or self.display_options.topic_enable:
                if not self.display_options.width > 0:
                    param_name = "Display width"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.display_options.width} must be a positive integer."
                    )
                if not self.display_options.height > 0:
                    param_name = "Display height"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.display_options.height} must be a positive integer."
                    )
                if not (
                    self.display_options.rate > 0
                    and self.display_options.rate <= self.simulation_frame_rate
                ):
                    param_name = "Display rate"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.display_options.rate} must be a positive number"
                        f" and <= than simulation frame rate {self.simulation_frame_rate}."
                    )

            # Check that the observation rates are lower than the simulation frame rate
            rate_param_err_msg = (
                "{param_name} {sensor_rate} must be non-negative and <="
                + " than simulation frame rate {sim_rate}".format(
                    sim_rate=self.simulation_frame_rate
                )
            )
            sensors = ["camera", "imu", "wheel_encoders"]
            for sensor in sensors:
                if getattr(self, f"{sensor}_obs_enable"):
                    if not (
                        0
                        < getattr(self, f"{sensor}_obs_rate")
                        <= self.simulation_frame_rate
                    ):
                        param_name = f"{sensor.capitalize()} observation rate"
                        list_error_param_names.append(param_name)
                        list_error_msgs.append(
                            rate_param_err_msg.format(
                                param_name=param_name,
                                sensor_rate=getattr(self, f"{sensor}_obs_rate"),
                            )
                        )

            # Check that the observation noise values are valid
            noise_param_err_msg = (
                "{param_name} {param_value} must be a non-negative number"
            )
            if self.imu_obs_enable and self.imu_noise.enable:
                IMU_fields = ["ang_vel", "lin_acc", "orientation"]
                noise_fields = ["white_noise_sigma", "bias_sigma"]
                for imu_field in IMU_fields:
                    for noise_field in noise_fields:
                        imu_field_attr = getattr(self.imu_noise, imu_field)
                        if imu_field_attr.enable:
                            noise_param = getattr(imu_field_attr, noise_field)
                            if not noise_param >= 0:
                                param_name = (
                                    f"IMU {imu_field} noise parameter {noise_field}"
                                )
                                list_error_param_names.append(param_name)
                                list_error_msgs.append(
                                    noise_param_err_msg.format(
                                        param_name=param_name,
                                        param_value=noise_param,
                                    )
                                )
            if self.wheel_encoders_obs_enable and (
                self.wheel_encoder_left_noise.enable
                or self.wheel_encoder_right_noise.enable
            ):
                wheel_encoders_names = ["left", "right"]
                noise_fields = ["white_noise_sigma"]
                for wheel_encoder_name in wheel_encoders_names:
                    for noise_field in noise_fields:
                        wheel_encoder_attr = getattr(
                            self, f"wheel_encoder_{wheel_encoder_name}_noise"
                        )
                        if wheel_encoder_attr.enable:
                            noise_param = getattr(wheel_encoder_attr, noise_field)
                            if not noise_param >= 0:
                                param_name = f"Wheel Encoder {wheel_encoder_name} noise parameter {noise_field}"
                                list_error_param_names.append(param_name)
                                list_error_msgs.append(
                                    noise_param_err_msg.format(
                                        param_name=param_name,
                                        param_value=noise_param,
                                    )
                                )

            if self.applied_command_noise.enable:
                noise_fields = ["white_noise_sigma", "bias_sigma"]
                for noise_field in noise_fields:
                    command_noise_param = getattr(
                        self.applied_command_noise, noise_field
                    )
                    if not command_noise_param >= 0:
                        param_name = f"Command noise parameter {noise_field}"
                        list_error_param_names.append(param_name)
                        list_error_msgs.append(
                            noise_param_err_msg.format(
                                param_name=param_name,
                                param_value=command_noise_param,
                            )
                        )
                if not (0 <= self.applied_command_noise_min_action_threshold <= 1):
                    param_name = "Applied command noise min action threshold"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        f"{param_name} {self.applied_command_noise_min_action_threshold} must be a number between 0 and 1."
                    )

            # Check that string parameters are valid
            string_param_err_msg = (
                "{param_name} {string_value} must be one of {accepted_strings}"
            )

            accepted_motion_models = ["dynamics", "kinematics"]
            if not (self.motion_model in accepted_motion_models):
                param_name = "Motion model"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(self.motion_model),
                        accepted_strings=logger.bold_green_on_black(
                            accepted_motion_models
                        ),
                    )
                )

            accepted_simulation_update_use = ["frame_rate", "delta_time_step"]
            if not (self.simulation_update_use in accepted_simulation_update_use):
                param_name = "Simulation update use"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name_name=param_name,
                        string_value=logger.bold_red_on_black(
                            self.simulation_update_use
                        ),
                        accepted_strings=logger.bold_green_on_black(
                            accepted_simulation_update_use
                        ),
                    )
                )

            accepted_duckiebot_colors = ["red", "blue"]
            if not (self.duckiebot_color in accepted_duckiebot_colors):
                param_name = "Duckiebot color"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(self.duckiebot_color),
                        accepted_strings=logger.bold_green_on_black(
                            accepted_duckiebot_colors
                        ),
                    )
                )

            accepted_map_names = self.available_maps
            if not (self.map_name in accepted_map_names):
                param_name = "Map name"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(self.map_name),
                        accepted_strings=logger.bold_green_on_black(accepted_map_names),
                    )
                )

            accepted_styles = [
                "photos",
                "photos-segmentation",
                "segmentation",
                "smooth",
                "synthetic",
            ]
            if not (self.style in accepted_styles):
                param_name = "Rendering style"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(self.style),
                        accepted_strings=logger.bold_green_on_black(accepted_styles),
                    )
                )

            accepted_reset_modes = ["reset", "kill"]
            if not (self.reset_mode in accepted_reset_modes):
                param_name = "Reset mode"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(self.reset_mode),
                        accepted_strings=logger.bold_green_on_black(
                            accepted_reset_modes
                        ),
                    )
                )

            accepted_camera_obs_compression_formats = ["jpeg", "png"]
            if not (
                self.camera_obs_compression_format
                in accepted_camera_obs_compression_formats
            ):
                param_name = "Camera compression format"
                list_error_param_names.append(param_name)
                list_error_msgs.append(
                    string_param_err_msg.format(
                        param_name=param_name,
                        string_value=logger.bold_red_on_black(
                            self.camera_obs_compression_format
                        ),
                        accepted_strings=logger.bold_green_on_black(
                            accepted_camera_obs_compression_formats
                        ),
                    )
                )

            if self.display_options.screen_enable or self.display_options.topic_enable:
                accepted_display_mode = ["human", "top_down", "free_camera"]
                if not (self.display_options.mode in accepted_display_mode):
                    param_name = "Display mode"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        string_param_err_msg.format(
                            param_name=param_name,
                            string_value=logger.bold_red_on_black(
                                self.display_options.mode
                            ),
                            accepted_strings=logger.bold_green_on_black(
                                accepted_display_mode
                            ),
                        )
                    )
                accepted_display_compression_formats = ["jpeg", "png"]
                if not (
                    self.display_options.compression_format
                    in accepted_display_compression_formats
                ):
                    param_name = "Display compression format"
                    list_error_param_names.append(param_name)
                    list_error_msgs.append(
                        string_param_err_msg.format(
                            param_name=param_name,
                            string_value=logger.bold_red_on_black(
                                self.display_options.compression_format
                            ),
                            accepted_strings=logger.bold_green_on_black(
                                accepted_display_compression_formats
                            ),
                        )
                    )

            if not Path(self.path_folder_mesh_resources).exists():
                param_name = "Mesh resources folder"
                # list_error_param_names.append(param_name)
                # list_error_msgs.append(
                #     f"{param_name} {self.path_folder_mesh_resources} does not exist."
                # )
                logger.warning(
                    f"{param_name} {self.path_folder_mesh_resources} does not exist."
                )

            if not Path(self.path_duckiebot_mesh_resource).exists():
                param_name = "Duckiebot mesh resource"
                # list_error_param_names.append(param_name)
                # list_error_msgs.append(
                #     f"{param_name} {self.path_duckiebot_mesh_resource} does not exist."
                # )
                logger.warning(
                    f"{param_name} {self.path_duckiebot_mesh_resource} does not exist."
                    " The duckiebot mesh will probably not be displayed in RViz."
                )

        except AttributeError as e:
            logger.critical(
                "Meta-error: The parameters check is not correctly implemented."
                " This should not happen. Maybe the class attributes have been renamed."
            )
            print()
            raise AttributeError from e

        if list_error_msgs:
            logger.error("Some parameters were not correctly set:")
            for error_msg in list_error_msgs:
                logger.error(f"  - {error_msg}")
            print("\n")
            raise ValueError(
                logger.red(
                    f"Some parameters were not correctly set: {list_error_param_names}."
                    + f" See above logging error messages for more details."
                )
            )

    def finish_setup_params(self) -> None:
        self.motion_model = MotionModelType[self.motion_model.upper()]

    def setup_tf_and_mesh(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.pub_duckiebot_mesh = rospy.Publisher(
            "duckiebot_marker", Marker, queue_size=10
        )
        self.duckiebot_marker = Marker()

        self.duckiebot_marker.header.frame_id = "base_link"
        self.duckiebot_marker.id = 0
        self.duckiebot_marker.ns = "duckiebot"
        self.duckiebot_marker.type = self.duckiebot_marker.MESH_RESOURCE
        self.duckiebot_marker.action = self.duckiebot_marker.ADD
        self.duckiebot_marker.mesh_resource = (
            f"file://{self.path_duckiebot_mesh_resource}"
        )
        self.duckiebot_marker.mesh_use_embedded_materials = True
        z_offset_mesh = 0.0335  # offset such that the wheels are on the ground
        self.duckiebot_marker.pose.position = Point(0, 0, z_offset_mesh)
        self.duckiebot_marker.pose.orientation = Quaternion(0, 0, 0, 1)
        self.duckiebot_marker.scale = Vector3(1, 1, 1)

    def setup_services(self) -> None:
        self.srv_server_reset_sim = rospy.Service(
            self.name_srv_server_reset_sim, Trigger, self.srv_server_reset_sim_callback
        )
        self.srv_server_kill_sim = rospy.Service(
            self.name_srv_server_kill_sim, Trigger, self.srv_server_kill_sim_callback
        )

    def setup_publishers(self) -> None:
        self.pub_camera = (
            rospy.Publisher(self.name_pub_camera_topic, CompressedImage, queue_size=10)
            if self.camera_obs_enable
            else None
        )
        self.pub_imu = (
            rospy.Publisher(self.name_pub_imu_topic, Imu, queue_size=10)
            if self.imu_obs_enable
            else None
        )
        self.pub_wheel_encoder_left = (
            rospy.Publisher(
                self.name_pub_wheel_encoder_left_topic,
                WheelEncoderStamped,
                queue_size=10,
            )
            if self.wheel_encoders_obs_enable
            else None
        )
        self.pub_wheel_encoder_right = (
            rospy.Publisher(
                self.name_pub_wheel_encoder_right_topic,
                WheelEncoderStamped,
                queue_size=10,
            )
            if self.wheel_encoders_obs_enable
            else None
        )

        self.pub_gt_pose = (
            rospy.Publisher(self.name_pub_gt_pose_topic, PoseStamped, queue_size=10)
            if self.gt_pose_obs_enable
            else None
        )

        self.pub_lane_pose = (
            rospy.Publisher(self.name_pub_gt_lane_pose_topic, LanePose, queue_size=10)
            if self.gt_lane_pose_obs_enable
            else None
        )

        self.pub_simulation_external_window = (
            rospy.Publisher(
                self.name_pub_external_window_topic, CompressedImage, queue_size=10
            )
            if self.display_options.topic_enable
            else None
        )

        self.pub_command = (
            rospy.Publisher(
                self.name_pub_command_topic, WheelsCmdStamped, queue_size=10
            )
            if self.command_obs_enable
            else None
        )

    def setup_subscribers(self) -> None:
        self.sub_wheel_cmd = rospy.Subscriber(
            self.name_sub_wheel_cmd_topic, WheelsCmdStamped, self.wheel_cmd_callback
        )

    @staticmethod
    def logger(level, txt) -> None:
        getattr(logger, level)(txt)

    def sleep(self) -> None:
        self.rospy_rate.sleep()

    def enable_test_mode(self) -> None:
        self.test_mode = True
        self.env.test_mode = True
        logger.info("Testing simulation is running...")
        logger.set_prefix_txt("TEST MODE ")

    def disable_test_mode(self) -> None:
        self.test_mode = False
        self.env.test_mode = False
        logger.set_prefix_txt("")
        logger.info("Testing finished.")
        self.print_info_sim_is_running()

    def srv_server_reset_sim_callback(self, req: Trigger) -> TriggerResponse:
        logger.info("Received service request to reset env.")
        self.prepare_to_reset_simulator()
        return TriggerResponse(success=True, message="Simulator will reset.")

    def srv_server_kill_sim_callback(self, req: Trigger) -> TriggerResponse:
        logger.info("Received service request to kill env.")
        self.prepare_to_kill_simulator()
        return TriggerResponse(success=True, message="Simulator will be killed.")

    def reset(self, force_reset_mode: Optional[str] = None) -> str:
        if (
            not self.test_mode
            and self.reset_mode == "kill"
            and force_reset_mode != "reset"
        ) or force_reset_mode == "kill":
            if force_reset_mode != "kill":
                info_txt = f"Called reset env with reset mode {logger.bold_green_on_black(self.reset_mode)}."
            else:
                info_txt = f"Called force kill env."
            logger.info(info_txt)
            self.prepare_to_kill_simulator()
            return "kill"

        # FIXME: this is a bugfix for a bug in the original duckietown-gym sim
        # the external window will be darker after a reset if
        # the render_img method is not called at least twice in a row.
        # Hence we call it again here before resetting the environment.
        if self.count_episodes > 0:
            self.env.render_obs()

        if self.reset_action_when_env_is_reset:
            self.action = np.array([0.0, 0.0])

        # FIXME: This is needed otherwise if the RSHIFT key was pressed
        # during the last step of previous episode, the key_handler
        # will not be reset and the robot will start moving at the beginning.
        # No idea why this is happening, so this is a workaround to manually
        # reset the key_handler values.
        for key_event in self.key_events_list:
            self.key_handler[key_event] = False

        self.env.reset(self.reset_ros_time_when_env_is_reset)
        logger.info("Environment reset.")

        if not self.test_mode:
            self.count_episodes += 1

        return "reset"

    def render_external(self) -> None:
        if self.display_options.screen_enable or (
            self.display_options.topic_enable
            and self.pub_simulation_external_window is not None
            and self.pub_simulation_external_window.get_num_connections() > 0
        ):
            img_simulation_external_window = self.env.render(
                render_on_screen=self.display_options.screen_enable,
                render_on_topic=self.display_options.topic_enable,
                mode=self.display_options.mode,
                segment=self.display_options.segmentation,
                info_enabled=self.display_options.info_enabled,
                info_pose=self.display_options.info_pose,
                info_speed=self.display_options.info_speed,
                info_steps=self.display_options.info_steps,
                info_time_stamp=self.display_options.info_time_stamp,
                reuse_camera_obs_if_possible=self.display_reuse_camera_obs_if_possible,
            )
            if (
                self.display_options.topic_enable
                and self.pub_simulation_external_window is not None
            ):

                image_msg = image_msg_from_cv_image(
                    img_simulation_external_window,
                    self.env.timestamp,
                    self.display_options.compression_format,
                )
                self.pub_simulation_external_window.publish(image_msg)

    def test_step(self, event=None) -> None:

        if self.called_kill_simulator:
            self.kill_simulator("Received kill signal.")
            return

        if self.called_reset_simulator:
            self.reset("reset")
            self.called_reset_simulator = False

        self.env.someone_listening_camera_obs = (
            self.pub_camera is not None and self.pub_camera.get_num_connections() > 0
        )

        # start_step = time.perf_counter()
        observations, reward, done, misc = self.env.step(np.array([0.0, 0.0]))
        # end_step = time.perf_counter()
        # print(f"Timing update step simulator: {end_step - start_step:.2g} s")

        self.observations = observations

        self.publish_observations(observations)

        if done:
            logger.info(f"{misc['Simulator']['msg']} Resetting environment.")
            self.reset()

        self.render_external()

    def step(self, event=None) -> None:

        if self.called_kill_simulator:
            self.kill_simulator("Received kill signal.")
            return

        if self.called_reset_simulator:
            self.reset("reset")
            self.called_reset_simulator = False

        # generate observation only if someone is subscribed to image topic, to save computation
        if self.pub_camera is not None:
            old_someone_listening_camera_obs = self.env.someone_listening_camera_obs
            self.env.someone_listening_camera_obs = (
                self.pub_camera.get_num_connections() > 0
            )
            if (
                not old_someone_listening_camera_obs
                and self.env.someone_listening_camera_obs
            ):
                logger.debug(
                    f"Someone subscribed to {self.name_pub_camera_topic}. Generating camera observations."
                )
            elif (
                old_someone_listening_camera_obs
                and not self.env.someone_listening_camera_obs
            ):
                logger.debug(
                    f"No one is subscribed to {self.name_pub_camera_topic}. Skipping camera observation generation."
                )

        # start_step = time.perf_counter()

        if (
            self.enable_additional_keyboard_control
            and self.display_options.screen_enable
        ):
            action_key_event, use_action_key_event = self.process_action_key_event()
        else:
            use_action_key_event = False

        # Keyboard control has priority over topic action
        if use_action_key_event:
            action = action_key_event.copy()
        else:
            action = self.action.copy()

        if self.applied_command_noise.enable:
            # add some randomness to the min action threshold
            applied_command_noise_min_action_threshold = (
                self.applied_command_noise_min_action_threshold
                + (
                    min(self.applied_command_noise_min_action_threshold / 2, 0.1)
                    * (2 * np.random.random() - 1)
                )
            )
            noise = np.array(
                [
                    self.applied_command_noise.sample(),
                    self.applied_command_noise.sample(update_bias=False),
                ]
            )  # sample anyway to update the noise mdoel
            if np.all(action < applied_command_noise_min_action_threshold):
                noise = np.array([0.0, 0.0])
            else:
                # 0 noise for abs(action) < threshold, max noise at max abs action
                noise_factor = (
                    1 + applied_command_noise_min_action_threshold
                ) * np.abs(action).mean() - applied_command_noise_min_action_threshold
                noise *= noise_factor
            applied_action = (action + noise).clip(-1, 1)
        else:
            applied_action = action.copy()

        observations, reward, done, misc = self.env.step(applied_action)
        # end_step = time.perf_counter()
        # print(f"Timing update step simulator: {end_step - start_step:.2g} s")

        # Overwrite the observed noisy action with the one the noisy-free input
        # Why this? Because we want to publish not the actual applied noisy action,
        # but the one that the robot thinks it is applying
        if observations.Command is not None:
            observations.Command.action_left = action[0]
            observations.Command.action_right = action[1]

        self.observations = observations

        if not self.test_mode:
            self.publish_observations(observations)
            self.broadcast_tf_and_mesh(
                observations.GroundTruthPose,
                observations.WheelEncoderLeft,
                observations.WheelEncoderRight,
            )

        if done:
            logger.info(f"{misc['Simulator']['msg']} Resetting environment.")
            self.reset()

        if self.env.step_count % (self.env.frame_rate / self.display_options.rate) < 1:
            self.render_external()

    def process_action_key_event(self) -> Tuple[np.ndarray, bool]:
        use_action_key_event = False

        # This action is [lin_vel, ang_vel] so it easy to understand how keys affect the motion,
        # then it is converted to [v_l, v_r]
        action_key_event = np.array([0.0, 0.0])

        for num_key_event in self.num_key_events_list:
            if self.key_handler[num_key_event]:
                action_key_event = np.array(
                    [DICT_NUM_KEY_EVENT_TO_LIN_ACTION[num_key_event], 0]
                )
                use_action_key_event = True
                break

        # Process UP arrow only if no num key event was pressed
        if self.key_handler[KeyEvent.UP] and not use_action_key_event:
            action_key_event += np.array([0.2, 0])
            use_action_key_event = True
        if self.key_handler[KeyEvent.DOWN]:
            action_key_event += np.array([-0.2, 0])
            use_action_key_event = True
        if self.key_handler[KeyEvent.LEFT]:
            action_key_event += np.array([0, 0.2 + action_key_event[0]])
            use_action_key_event = True
        if self.key_handler[KeyEvent.RIGHT]:
            action_key_event += np.array([0, -0.2 - action_key_event[0]])
            use_action_key_event = True
        if self.key_handler[KeyEvent.SPACE]:
            action_key_event = np.array([0, 0])
            use_action_key_event = True
        if self.key_handler[KeyEvent.RSHIFT]:
            action_key_event[1] *= 1.5
            use_action_key_event = True

        if use_action_key_event:

            if self.env.is_delay_dynamics:
                parameters = self.env.state.state.parameters
            else:
                parameters = self.env.state.parameters

            wheel_distance = parameters.wheel_distance
            wheel_radius_left = parameters.wheel_radius_left
            wheel_radius_right = parameters.wheel_radius_right

            max_lin_vel = min(
                1,
                max(
                    -1.0,
                    MAX_WHEEL_ANG_VEL * (wheel_radius_right + wheel_radius_left) / 2,
                ),
            )
            max_ang_vel = min(
                6,
                max(
                    -6,
                    MAX_WHEEL_ANG_VEL
                    * (wheel_radius_right + wheel_radius_left)
                    / wheel_distance,
                ),
            )

            v1 = action_key_event[0] * max_lin_vel
            v2 = action_key_event[1] * max_ang_vel

            omega_r = (v1 + 0.5 * v2 * wheel_distance) / wheel_radius_right
            omega_l = (v1 - 0.5 * v2 * wheel_distance) / wheel_radius_left

            # conversion from motor rotation rate to duty cycle
            u_r = omega_r / MAX_WHEEL_ANG_VEL
            u_l = omega_l / MAX_WHEEL_ANG_VEL

            # limiting output to limit, which is 1.0 for the duckiebot
            u_r_limited = max(min(u_r, 1), -1)
            u_l_limited = max(min(u_l, 1), -1)

            action_key_event = np.array([u_l_limited, u_r_limited])

        return action_key_event, use_action_key_event

    def broadcast_tf_and_mesh(
        self,
        gt_pose: Optional[PoseObservation],
        wheel_encoder_left: Optional[WheelEncoderObservation],
        wheel_encoder_right: Optional[WheelEncoderObservation],
    ) -> None:

        # Broadcast the transform from world to robot frame
        if gt_pose:
            t_robot_frame = TransformStamped()

            t_robot_frame.header.stamp = rospy.Time(gt_pose.header.stamp)
            t_robot_frame.header.frame_id = gt_pose.header.frame_id
            t_robot_frame.child_frame_id = "base_link"
            t_robot_frame.transform.translation = Vector3(*gt_pose.position)
            t_robot_frame.transform.rotation = Quaternion(*gt_pose.orientation)

            self.tf_broadcaster.sendTransform(t_robot_frame)

        self.pub_duckiebot_mesh.publish(self.duckiebot_marker)

        # Broadcast the transform from robot to wheel_left and wheel_right frame
        if wheel_encoder_left:
            t_wheel_left_frame = TransformStamped()

            t_wheel_left_frame.header.stamp = rospy.Time(
                wheel_encoder_left.header.stamp
            )
            t_wheel_left_frame.header.frame_id = "robot_frame"
            t_wheel_left_frame.child_frame_id = wheel_encoder_left.header.frame_id
            t_wheel_left_frame.transform.translation = Vector3(
                0, self.env.wheel_dist / 2, 0
            )
            quaternion = tf.transformations.quaternion_from_euler(
                0, wheel_encoder_left.ticks * self.env.wheel_encoders_resolution_rad, 0
            )
            t_wheel_left_frame.transform.rotation = Quaternion(*quaternion)

            self.tf_broadcaster.sendTransform(t_wheel_left_frame)

        if wheel_encoder_right:
            t_wheel_right_frame = TransformStamped()

            t_wheel_right_frame.header.stamp = rospy.Time(
                wheel_encoder_right.header.stamp
            )
            t_wheel_right_frame.header.frame_id = "robot_frame"
            t_wheel_right_frame.child_frame_id = wheel_encoder_right.header.frame_id
            t_wheel_right_frame.transform.translation = Vector3(
                0, -self.env.wheel_dist / 2, 0
            )
            quaternion = tf.transformations.quaternion_from_euler(
                0, wheel_encoder_right.ticks * self.env.wheel_encoders_resolution_rad, 0
            )
            t_wheel_right_frame.transform.rotation = Quaternion(*quaternion)

            self.tf_broadcaster.sendTransform(t_wheel_right_frame)

    def publish_observations(self, observations: Observations) -> None:
        camera_obs = observations.Camera
        imu_obs = observations.Imu
        wheel_encoder_left_obs = observations.WheelEncoderLeft
        wheel_encoder_right_obs = observations.WheelEncoderRight
        command_obs = observations.Command
        gt_pose_obs = observations.GroundTruthPose
        gt_lane_pose_obs = observations.GroundTruthLanePose

        if self.pub_camera and camera_obs:
            image_msg = image_msg_from_obs(
                camera_obs, self.camera_obs_compression_format
            )
            self.pub_camera.publish(image_msg)

        if self.pub_imu and imu_obs:
            imu_msg = imu_msg_from_obs(imu_obs)
            self.pub_imu.publish(imu_msg)

        if self.pub_wheel_encoder_left and wheel_encoder_left_obs:
            wheel_encoder_left_msg = wheel_encoder_msg_from_obs(wheel_encoder_left_obs)
            self.pub_wheel_encoder_left.publish(wheel_encoder_left_msg)

        if self.pub_wheel_encoder_right and wheel_encoder_right_obs:
            wheel_encoder_right_msg = wheel_encoder_msg_from_obs(
                wheel_encoder_right_obs
            )
            self.pub_wheel_encoder_right.publish(wheel_encoder_right_msg)

        if self.pub_command and command_obs:
            wheel_cmd_msg = wheels_cmd_msg_from_command_obs(command_obs)
            self.pub_command.publish(wheel_cmd_msg)

        if self.gt_pose_obs_enable and gt_pose_obs:
            gt_pose_msg = pose_msg_from_pose_obs(gt_pose_obs)
            self.pub_gt_pose.publish(gt_pose_msg)

        if self.gt_lane_pose_obs_enable and gt_lane_pose_obs:
            gt_lane_pose_msg = lane_pose_msg_from_lane_pose_obs(gt_lane_pose_obs)
            self.pub_lane_pose.publish(gt_lane_pose_msg)

    def wheel_cmd_callback(self, msg) -> None:
        if not self.test_mode:
            self.action = np.array([msg.vel_left, msg.vel_right])

            logger.info(
                f"Received action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
            )

            if not self.repeat_action_until_new:
                self.action = np.array([0, 0])
