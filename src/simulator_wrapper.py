#!/usr/bin/env python3

from typing import Tuple, Dict, Optional
import numpy as np
from gym import spaces
import time
import os
import cv2
from ctypes import POINTER

import pyglet
from pyglet import gl, image, window

import geometry

from . import logger

from .simulator import Simulator

from gym_duckietown import list_maps2

from .simulator import (
    get_agent_corners,
    _actual_center,
    get_dir_vec,
    get_right_vec,
    DoneRewardInfo,
    MotionModelType,
    LanePosition,
    MAX_WHEEL_ANG_VEL,
    REWARD_INVALID_POSE,
    ROBOT_HEIGHT,
    ROBOT_WIDTH,
    ROBOT_LENGTH,
    WHEEL_DIST,
    NotInLane,
)

from .utils_ros import get_quaternion_from_yaw

from .custom_types import NumberType

from .Observations import (
    Observations,
    CameraObservation,
    ImuObservation,
    WheelEncoderObservation,
    CommandObservation,
    PoseObservation,
    LanePoseObservation,
    HeaderObservation,
    ImuNoise,
    WheelEncoderNoise,
    ENCODER_TYPE_ABSOLUTE,
    ENCODER_TYPE_INCREMENTAL,
)


class SimulatorWrapper(Simulator):

    def __init__(
        self,
        camera_obs_enable: bool,
        imu_obs_enable: bool,
        wheel_encoders_obs_enable: bool,
        command_obs_enable: bool,
        gt_pose_obs_enable: bool,
        gt_lane_pose_obs_enable: bool,
        camera_obs_rate: NumberType,
        imu_obs_rate: NumberType,
        wheel_encoders_obs_rate: NumberType,
        command_obs_rate: NumberType,
        gt_lane_pose_obs_rate: NumberType,
        gt_pose_obs_rate: NumberType,
        wheel_encoders_resolution: int,
        imu_noise: ImuNoise,
        wheel_encoder_left_noise: WheelEncoderNoise,
        wheel_encoder_right_noise: WheelEncoderNoise,
        check_drivable_enable: bool,
        check_collisions_enable: bool,
        compute_reward_enable: bool,
        seed: int,
        **kwargs,
    ):

        self.test_mode = True

        np.random.seed(seed)

        # debug
        self.ext_window_total_render_timer = [0, 0]
        self.camera_obs_total_render_time = [0, 0]

        self.check_drivable_enable = check_drivable_enable
        self.check_collisions_enable = check_collisions_enable

        self.compute_reward_enable = compute_reward_enable

        super().__init__(
            seed=seed,
            motion_model_encoder_resolution_rad=(2 * np.pi / wheel_encoders_resolution),
            **kwargs,
        )
        # pyglet_config = pyglet.gl.Config(
        #     sample_buffers=1, samples=4, depth_size=24, double_buffer=True
        # )
        # self.shadow_window = pyglet.window.Window(
        #     width=1, height=1, visible=False, config=pyglet_config
        # )

        self.cumulative_ros_time = 0

        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=13,
            x=5,
            y=self.window_height - 19,
            color=(245, 245, 245, 255),
        )

        if self.is_delay_dynamics:
            self.prev_linear_vel = geometry.linear_angular_from_se2(
                self.state.state.v0
            )[0]
        else:
            self.prev_linear_vel = geometry.linear_angular_from_se2(self.state.v0)[0]

        self.camera_obs_enable = camera_obs_enable
        self.imu_obs_enable = imu_obs_enable
        self.wheel_encoders_obs_enable = wheel_encoders_obs_enable
        self.command_obs_enable = command_obs_enable
        self.gt_pose_obs_enable = gt_pose_obs_enable
        self.gt_lane_pose_obs_enable = gt_lane_pose_obs_enable

        self.someone_listening_camera_obs = True

        self.camera_obs_rate = camera_obs_rate
        self.imu_obs_rate = imu_obs_rate
        self.wheel_encoders_obs_rate = wheel_encoders_obs_rate
        self.command_obs_rate = command_obs_rate
        self.gt_pose_obs_rate = gt_pose_obs_rate
        self.gt_lane_pose_obs_rate = gt_lane_pose_obs_rate

        self.wheel_encoders_resolution = wheel_encoders_resolution
        self.wheel_encoders_resolution_rad = 2 * np.pi / self.wheel_encoders_resolution

        self.imu_noise = imu_noise
        self.wheel_encoder_left_noise = wheel_encoder_left_noise
        self.wheel_encoder_right_noise = wheel_encoder_right_noise

        self.seq_camera_obs = 0
        self.seq_imu_obs = 0
        self.seq_wheel_encoders_obs = 0
        self.seq_command_obs = 0
        self.seq_gt_pose_obs = 0
        self.seq_gt_lane_pose_obs = 0

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.last_observation = [-1, None]

    def __del__(self):
        self.close()

    @staticmethod
    def logger(level, txt):
        getattr(logger, level)(txt)

    def step(self, action) -> Tuple[Observations, float, bool, Dict]:

        vels = action

        obs, reward, done, info = self._step(vels)

        return obs, reward, done, info

    def reset(self, reset_time: bool = True) -> None:
        self.last_observation = [-1, None]
        if not reset_time:
            self.cumulative_ros_time = self.timestamp
        super().reset()
        if not reset_time:
            self.timestamp = self.cumulative_ros_time
        if self.is_delay_dynamics:
            self.prev_linear_vel = geometry.linear_angular_from_se2(
                self.state.state.v0
            )[0]
        else:
            self.prev_linear_vel = geometry.linear_angular_from_se2(self.state.v0)[0]

    # Overriding 'step' function with adapted version
    def _step(self, action: np.ndarray) -> Tuple[Observations, float, bool, Dict]:
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        # for _ in range(self.frame_skip):
        delta_time = 1.0 / self.frame_rate
        self.update_physics(action, delta_time=delta_time)
        self.last_used_delta_time = delta_time

        obs = self.generate_observations()

        misc = self.get_agent_info()

        is_valid_pose = self.check_valid_pose(print_info=True)

        d = self.compute_done_reward(is_valid_pose)
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc

    def check_valid_pose(self, print_info=False) -> bool:
        return self._valid_pose(self.cur_pos, self.cur_angle, print_info)

    def compute_done_reward(self, is_valid_pose: bool) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        if not is_valid_pose:
            msg = "Stopping the simulator because we are at an invalid pose."
            # logger.info(msg)
            reward = REWARD_INVALID_POSE
            done_code = "invalid-pose"
            done = True
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = (
                "Stopping the simulator because we reached max_steps = %s"
                % self.max_steps
            )
            # logger.info(msg)
            done = True
            reward = 0
            done_code = "max-steps-reached"
        elif self.compute_reward_enable:
            done = False
            reward = self.compute_reward(self.cur_pos, self.cur_angle, self.robot_speed)
            msg = ""
            done_code = "in-progress"
        else:
            done = not is_valid_pose
            reward = 0
            msg = "No reward computation."
            done_code = "no-reward-computation"
        return DoneRewardInfo(
            done=done, done_why=msg, reward=reward, done_code=done_code
        )

    def _valid_pose(
        self,
        pos: geometry.T3value,
        angle: float,
        safety_factor: float = 1.0,
        print_info: bool = False,
    ) -> bool:
        """
        Check that the agent is in a valid pose

        safety_factor = minimum distance
        """

        res = True

        # Compute the coordinates of the base of both wheels
        if self.check_drivable_enable or self.check_collisions_enable:
            pos = _actual_center(pos, angle)

        if self.check_drivable_enable:
            if not self.drivable_tiles:
                logger.warning(
                    "Checking for duckiebot on drivable area"
                    " but no drivable tiles in the map"
                )
            f_vec = get_dir_vec(angle)
            r_vec = get_right_vec(angle)

            l_pos = pos - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
            r_pos = pos + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
            f_pos = pos + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

            # Check that the center position and
            # both wheels are on drivable tiles and no collisions

            all_drivable = (
                self._drivable_pos(pos)
                and self._drivable_pos(l_pos)
                and self._drivable_pos(r_pos)
                and self._drivable_pos(f_pos)
            )
            res = res and all_drivable

        # Recompute the bounding boxes (BB) for the agent
        if self.check_collisions_enable:
            agent_corners = get_agent_corners(pos, angle)
            no_collision = not self._collision(agent_corners)
            res = res and no_collision

        # if not res:
        #     logger.debug(
        #         f"Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}"
        #     )
        #     logger.debug(f"safety_factor: {safety_factor}")
        #     logger.debug(f"pos: {pos}")
        #     logger.debug(f"l_pos: {l_pos}")
        #     logger.debug(f"r_pos: {r_pos}")
        #     logger.debug(f"f_pos: {f_pos}")

        if print_info and (
            (self.check_drivable_enable and not all_drivable)
            or (self.check_collisions_enable and not no_collision)
        ):
            info_txt = "Invalid pose: "
            if self.check_drivable_enable and not all_drivable:
                info_txt += "not on drivable area"
                if self.check_collisions_enable and not no_collision:
                    info_txt += " and"
            if self.check_collisions_enable and not no_collision:
                info_txt += "collision detected"
            logger.info(info_txt)

        return res

    def render(
        self,
        render_on_screen: bool = False,
        render_on_topic: bool = False,
        mode: str = "human",
        info_enabled: bool = False,
        info_pose: bool = False,
        info_speed: bool = False,
        info_steps: bool = False,
        info_time_stamp: bool = False,
        close: bool = False,
        segment: bool = False,
        reuse_camera_obs_if_possible: bool = True,
    ):
        """
        Render the environment for human viewing

        mode: "human", "top_down", "free_cam", "rgb_array"

        """
        assert mode in ["human", "top_down", "free_cam", "rgb_array"]

        if close:
            if self.window:
                self.window.close()
            return

        top_down = mode == "top_down"
        # Render the image

        start_timer = time.perf_counter()
        do_render_image = True
        # Check if the camera observation rendering can be reused
        if reuse_camera_obs_if_possible:
            if not top_down and self.last_observation[0] == self.step_count:
                last_obs = self.last_observation[1]
                if last_obs.shape[:2] != (self.window_height, self.window_width):
                    if np.isclose(
                        last_obs.shape[0] / last_obs.shape[1],
                        self.window_height / self.window_width,
                    ):
                        last_obs = cv2.resize(
                            last_obs,
                            (self.window_width, self.window_height),
                            interpolation=(
                                cv2.INTER_NEAREST
                                if self.window_height > 1.5 * last_obs.shape[0]
                                else cv2.INTER_LINEAR
                            ),
                        )
                        do_render_image = False
                    else:
                        do_render_image = True
                else:
                    do_render_image = False
            else:
                do_render_image = True

        if do_render_image:
            img = self._render_img(
                self.window_width,
                self.window_height,
                self.multi_fbo_human,
                self.final_fbo_human,
                self.img_array_human,
                top_down=top_down,
                segment=segment,
            )
        else:
            img = last_obs

        end_timer = time.perf_counter()
        self.ext_window_total_render_timer[0] += end_timer - start_timer
        self.ext_window_total_render_timer[1] += 1
        print_render_time_every_k_steps = 200
        if self.ext_window_total_render_timer[1] % print_render_time_every_k_steps == 0:
            logger.debug_timing(
                f"Mean ext. window render time last {print_render_time_every_k_steps} steps: "
                f" {1000*(self.ext_window_total_render_timer[0]/self.ext_window_total_render_timer[1]):.2f} ms"
            )
            self.ext_window_total_render_timer = [0, 0]

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort and mode != "free_cam":
            img = self.camera_model.distort(img)

        if mode == "rgb_array":
            return img

        x, y, z = self.cur_pos[0], -self.cur_pos[2], self.cur_pos[1]
        info_text_list = [
                (
                    f"pose: ({x:.2f}, {y:.2f}), {self.cur_angle:.2f} rad, "
                    if info_pose
                    else ""
                ),
                f"speed: {self.speed:.2f} m/s, " if info_speed else "",
                f"steps: {self.step_count}, " if info_steps else "",
                f"time: {self.timestamp:.2f} s" if info_time_stamp else "",
        ]

        if render_on_screen:

            if self.window is None:
                config = gl.Config(double_buffer=False)
                self.window = window.Window(
                    width=self.window_width,
                    height=self.window_height,
                    resizable=False,
                    config=config,
                )

            start_timer_window = time.perf_counter()
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()

            timer1 = time.perf_counter()

            # Bind the default frame buffer
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            # Setup orghogonal projection
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            gl.glOrtho(0, self.window_width, 0, self.window_height, 0, 10)

            # # Draw the image to the rendering window
            width = img.shape[1]
            height = img.shape[0]
            flipped_img = np.ascontiguousarray(np.flip(img, axis=0))

            timer2 = time.perf_counter()

            img_data = image.ImageData(
                width,
                height,
                "RGB",
                flipped_img.ctypes.data_as(POINTER(gl.GLubyte)),
                pitch=width * 3,
            )
            timer3 = time.perf_counter()

            img_data.blit(0, 0, 0, width=self.window_width, height=self.window_height)

            timer4 = time.perf_counter()

            # Display position/state information
            if mode != "free_cam" and info_enabled:
                self.text_label.text = "".join(info_text_list)
                self.text_label.draw()

            timer5 = time.perf_counter()

            # Force execution of queued commands
            gl.glFlush()

            end_timer_window = time.perf_counter()

            # print(f"TOTAL Render time window: {1000*(end_timer_window - start_timer_window):.2f} ms")
            # print(f"  wind : {1000*(timer1 - start_timer_window):.2f} ms")
            # print(f"  set  : {1000*(timer2 - timer1):.2f} ms")
            # print(f"  img  : {1000*(timer3 - timer2):.2f} ms")
            # print(f"  blit : {1000*(timer4 - timer3):.2f} ms")
            # print(f"  text : {1000*(timer5 - timer4):.2f} ms")
            # print(f"  flush: {1000*(end_timer_window - timer5):.2f} ms")

        if render_on_topic:
            u, v = 10, 0
            if mode != "free_cam" and info_enabled:
                font_type = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_thickness = 1
                for info_text_element in info_text_list:
                    if info_text_element == "":
                        continue
                    # compute the position of the text
                    (dy, dx), baseline = cv2.getTextSize(
                        info_text_element,
                        font_type,
                        font_scale,
                        font_thickness,
                    )
                    v += dx + baseline
                    cv2.putText(
                        img,
                        info_text_element,
                        (u, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness,
                        cv2.LINE_AA,
                    )

        return img

    def generate_observations(self) -> Observations:
        # Generate the current camera image
        if (
            self.camera_obs_enable
            and self.step_count % (self.frame_rate / self.camera_obs_rate) < 1
        ):

            if self.someone_listening_camera_obs:
                camera_obs = self.generate_camera_obs()
            else:
                camera_obs = None
        else:
            camera_obs = None

        # Generate the current IMU data
        if (
            self.imu_obs_enable
            and self.step_count % (self.frame_rate / self.imu_obs_rate) < 1
        ):
            imu_obs = self.generate_imu_obs()
        else:
            if self.imu_noise.enable:  # sample anyway the noise to update the bias
                self.imu_noise.lin_acc.sample()
                self.imu_noise.ang_vel.sample()
                self.imu_noise.orientation.sample()
            imu_obs = None

        # Generate the current wheel encoder data
        if (
            self.wheel_encoders_obs_enable
            and self.step_count % (self.frame_rate / self.wheel_encoders_obs_rate) < 1
        ):
            wheel_encoder_left_obs, wheel_encoder_right_obs = (
                self.generate_wheel_encoders_obs()
            )
        else:
            if (
                self.wheel_encoder_left_noise.enable
            ):  # sample anyway the noise to update the bias
                self.wheel_encoder_left_noise.sample()
            if (
                self.wheel_encoder_right_noise.enable
            ):  # sample anyway the noise to update the bias
                self.wheel_encoder_right_noise.sample()
            wheel_encoder_left_obs = None
            wheel_encoder_right_obs = None

        # Generate the current command data
        if (
            self.command_obs_enable
            and self.step_count % (self.frame_rate / self.command_obs_rate) < 1
        ):
            command_obs = self.generate_command_obs()
        else:
            command_obs = None

        # Generate the current ground truth global pose
        if (
            self.gt_pose_obs_enable
            and self.step_count % (self.frame_rate / self.gt_pose_obs_rate) < 1
        ):
            gt_pose_obs = self.generate_pose_obs()
        else:
            gt_pose_obs = None

        # Generate the current ground truth relative lane pose
        if (
            self.gt_lane_pose_obs_enable
            and self.step_count % (self.frame_rate / self.gt_lane_pose_obs_rate) < 1
        ):
            gt_lane_pose_obs = self.generate_lane_pose_obs()
        else:
            gt_lane_pose_obs = None

        return Observations(
            Camera=camera_obs,
            Imu=imu_obs,
            WheelEncoderLeft=wheel_encoder_left_obs,
            WheelEncoderRight=wheel_encoder_right_obs,
            Command=command_obs,
            GroundTruthPose=gt_pose_obs,
            GroundTruthLanePose=gt_lane_pose_obs,
        )

    def generate_camera_obs(self) -> CameraObservation:
        start_timing = time.time()
        camera_image = self.render_obs()
        end_timing = time.time()

        if not self.test_mode:
            self.camera_obs_total_render_time[0] += end_timing - start_timing
            self.camera_obs_total_render_time[1] += 1
            print_render_time_every_k_steps = 200
            if (
                self.camera_obs_total_render_time[1] % print_render_time_every_k_steps
                == 0
            ):
                logger.debug_timing(
                    f"Mean camera obs render time last {print_render_time_every_k_steps} steps: "
                    f" {1000*(self.camera_obs_total_render_time[0]/self.camera_obs_total_render_time[1]):.2f} ms"
                )
                self.camera_obs_total_render_time = [0, 0]

        self.last_observation = [self.step_count, camera_image]

        header_camera_obs = HeaderObservation(
            seq=self.seq_camera_obs, stamp=self.timestamp, frame_id="camera_frame"
        )
        camera_obs = CameraObservation(header=header_camera_obs, image=camera_image)

        self.seq_camera_obs += 1

        return camera_obs

    def generate_imu_obs(self) -> ImuObservation:

        if self.is_delay_dynamics:
            linear_angular = geometry.linear_angular_from_se2(self.state.state.v0)
        else:
            linear_angular = geometry.linear_angular_from_se2(self.state.v0)
        longit, lateral = linear_angular[0]
        angular = linear_angular[1]

        # angular velocity
        angular += self.imu_noise.ang_vel.sample()
        ang_vel = np.array([0, 0, angular])
        ang_vel_covariance = np.zeros((9,))
        ang_vel_covariance[8] = self.imu_noise.ang_vel.get_nominal_variance()

        # linear acceleration
        longit_prev, lateral_prev = self.prev_linear_vel
        lin_acc = (
            np.array([longit - longit_prev, lateral - lateral_prev, 0])
            / self.last_used_delta_time
        )
        lin_acc_x_noise = self.imu_noise.lin_acc.sample()
        # To use one noise object instance, call it again but disable the bias update
        lin_acc_y_noise = self.imu_noise.lin_acc.sample(update_bias=False)

        lin_acc[0] += lin_acc_x_noise
        lin_acc[1] += lin_acc_y_noise
        lin_acc_covariance = np.zeros((9,))
        lin_acc_covariance[0] = self.imu_noise.lin_acc.get_nominal_variance()
        lin_acc_covariance[4] = self.imu_noise.lin_acc.get_nominal_variance()

        # orientation
        # This is a simplification: instead of integrating the angular velocity to get the orientation,
        # which is how the IMU would work in reality and hence dependent on the angular velocity measurement,
        # we just use the current angle provided by the simulation as the orientation
        orientation_tmp = self.cur_angle
        orientation_tmp += self.imu_noise.orientation.sample()
        orientation = get_quaternion_from_yaw(orientation_tmp)
        orientation_covariance = np.zeros((9,))
        orientation_covariance[8] = self.imu_noise.orientation.get_nominal_variance()

        self.prev_linear_vel = [longit, lateral]

        header_imu_obs = HeaderObservation(
            seq=self.seq_imu_obs, stamp=self.timestamp, frame_id="imu_frame"
        )

        imu_obs = ImuObservation(
            header=header_imu_obs,
            angular_velocity=ang_vel,
            angular_velocity_covariance=ang_vel_covariance,
            linear_acceleration=lin_acc,
            linear_acceleration_covariance=lin_acc_covariance,
            orientation=orientation,
            orientation_covariance=orientation_covariance,
        )

        self.seq_imu_obs += 1

        return imu_obs

    def generate_wheel_encoders_obs(self) -> WheelEncoderObservation:

        wheel_encoder_type = ENCODER_TYPE_INCREMENTAL

        left_resolution = self.wheel_encoders_resolution
        right_resolution = self.wheel_encoders_resolution

        if self.is_delay_dynamics:
            axis_left_rad = self.state.state.axis_left_rad
            axis_right_rad = self.state.state.axis_right_rad
        else:
            axis_left_rad = self.state.axis_left_rad
            axis_right_rad = self.state.axis_right_rad

        axis_left_rad += self.wheel_encoder_left_noise.sample()
        axis_right_rad += self.wheel_encoder_right_noise.sample()

        if wheel_encoder_type == ENCODER_TYPE_ABSOLUTE:
            axis_left_rad = axis_left_rad % (2 * np.pi)
            axis_right_rad = axis_right_rad % (2 * np.pi)

        left_ticks = int(np.round(axis_left_rad / self.wheel_encoders_resolution_rad))
        right_ticks = int(np.round(axis_right_rad / self.wheel_encoders_resolution_rad))

        if left_ticks < -2147483648:
            left_ticks = -2147483648
            logger.warning("Left wheel encoder ticks underflowed.")
        elif left_ticks > 2147483647:
            left_ticks = 2147483647
            logger.warning("Left wheel encoder ticks overflowed.")
        if right_ticks < -2147483648:
            right_ticks = -2147483648
            logger.warning("Right wheel encoder ticks underflowed.")
        elif right_ticks > 2147483647:
            right_ticks = 2147483647
            logger.warning("Right wheel encoder ticks overflowed.")

        header_wheel_encoders_left_obs = HeaderObservation(
            seq=self.seq_wheel_encoders_obs,
            stamp=self.timestamp,
            frame_id="wheel_left_frame",
        )
        wheel_encoder_left_obs = WheelEncoderObservation(
            header=header_wheel_encoders_left_obs,
            ticks=left_ticks,
            resolution=left_resolution,
            type=wheel_encoder_type,
        )

        header_wheel_encoders_right_obs = HeaderObservation(
            seq=self.seq_wheel_encoders_obs,
            stamp=self.timestamp,
            frame_id="wheel_right_frame",
        )
        wheel_encoder_right_obs = WheelEncoderObservation(
            header=header_wheel_encoders_right_obs,
            ticks=right_ticks,
            resolution=right_resolution,
            type=wheel_encoder_type,
        )

        self.seq_wheel_encoders_obs += 1

        return wheel_encoder_left_obs, wheel_encoder_right_obs

    def generate_command_obs(self) -> CommandObservation:
        header_command_obs = HeaderObservation(
            seq=self.seq_command_obs,
            stamp=self.timestamp,
            frame_id="",
        )
        command_obs = CommandObservation(
            header=header_command_obs,
            action_left=self.last_action[0],
            action_right=self.last_action[1],
        )

        self.seq_command_obs += 1

        return command_obs

    def generate_pose_obs(self) -> PoseObservation:
        header_gt_pose_obs = HeaderObservation(
            seq=self.seq_gt_pose_obs, stamp=self.timestamp, frame_id="world"
        )
        gt_pose_obs = PoseObservation(
            header=header_gt_pose_obs,
            position=[self.cur_pos[0], -self.cur_pos[2], self.cur_pos[1]],
            orientation=get_quaternion_from_yaw(self.cur_angle),
        )

        self.seq_gt_pose_obs += 1

        return gt_pose_obs

    def generate_lane_pose_obs(self) -> LanePoseObservation:

        try:
            lp = self.get_lane_pos2(self.cur_pos, self.cur_angle)
        except NotInLane:
            return None

        header_gt_lane_pose_obs = HeaderObservation(
            seq=self.seq_gt_lane_pose_obs, stamp=self.timestamp, frame_id="world"
        )
        gt_lane_pose_obs = LanePoseObservation(
            header=header_gt_lane_pose_obs, d=lp.dist, phi=lp.angle_rad
        )

        self.seq_gt_lane_pose_obs += 1

        return gt_lane_pose_obs
