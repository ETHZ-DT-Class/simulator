#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .custom_types import (
    ImageType,
    Vector2Type,
    Vector3Type,
    QuaternionType,
    Covariance3Type,
)


ENCODER_TYPE_ABSOLUTE = 0
ENCODER_TYPE_INCREMENTAL = 1


@dataclass
class HeaderObservation:
    seq: int
    stamp: float
    frame_id: str


@dataclass
class CameraObservation:
    header: HeaderObservation
    image: ImageType


@dataclass
class ImuObservation:
    header: HeaderObservation
    angular_velocity: Vector3Type
    angular_velocity_covariance: Covariance3Type
    linear_acceleration: Vector3Type
    linear_acceleration_covariance: Covariance3Type
    orientation: QuaternionType
    orientation_covariance: Covariance3Type


@dataclass
class WheelEncoderObservation:
    header: HeaderObservation
    ticks: int
    resolution: int
    type: int


@dataclass
class PoseObservation:
    header: HeaderObservation
    position: Vector3Type
    orientation: QuaternionType


@dataclass
class CommandObservation:
    header: HeaderObservation
    action_left: float
    action_right: float


@dataclass
class LanePoseObservation:
    header: HeaderObservation
    d: float
    phi: float


@dataclass
class Observations:
    Camera: Optional[CameraObservation] = None
    Imu: Optional[ImuObservation] = None
    WheelEncoderLeft: Optional[WheelEncoderObservation] = None
    WheelEncoderRight: Optional[WheelEncoderObservation] = None
    Command: Optional[CommandObservation] = None
    GroundTruthPose: Optional[PoseObservation] = None
    GroundTruthLanePose: Optional[LanePoseObservation] = None


@dataclass
class Noise:
    enable: bool
    white_noise_sigma: float
    bias_mu: float
    bias_sigma: float

    def __post_init__(self):
        self.white_noise_variance = self.white_noise_sigma**2

    def __str__(self) -> str:
        return (
            f"{'EN' if self.enable else 'DIS'}ABLED"
            f"{f' white_noise_sigma: {self.white_noise_sigma}, bias_mu: {self.bias_mu}, bias_sigma: {self.bias_sigma}' if self.enable else ''}"
        )

    def sample(self, update_bias=True) -> float:
        if self.enable:
            if update_bias and self.bias_sigma > 0:
                self.bias_mu += self._sample_from_limited_normal(self.bias_sigma)
            if self.white_noise_sigma > 0:
                white_noise = self._sample_from_limited_normal(self.white_noise_sigma)
            else:
                white_noise = 0.0
            return self.bias_mu + white_noise
        else:
            return 0.0

    def _sample_from_limited_normal(
        self, sigma: float, sigma_factor_limit: float = 5.0
    ) -> float:
        sampled = np.random.normal(0, sigma)
        return np.clip(sampled, -sigma_factor_limit * sigma, sigma_factor_limit * sigma)

    def get_nominal_variance(self) -> float:
        # TODO return variance of the white noise only, or
        # the sum of the white noise and bias variance? The latter
        # is more precise, but it should be unknown to the filter.
        # return self.white_noise_variance + self.bias_sigma**2
        return self.white_noise_variance


@dataclass
class ImuNoise:
    lin_acc: Noise
    ang_vel: Noise
    orientation: Noise

    def __post_init__(self):
        self.enable = (
            self.lin_acc.enable or self.ang_vel.enable or self.orientation.enable
        )

    def __str__(self) -> str:
        return (
            f"lin_acc: {self.lin_acc}\n"
            f"ang_vel: {self.ang_vel}\n"
            f"orientation: {self.orientation}"
        )


@dataclass
class WheelEncoderNoise(Noise):
    pass


@dataclass
class AppliedCommandNoise(Noise):
    pass
