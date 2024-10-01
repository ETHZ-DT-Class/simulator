import copy
from enum import Enum, IntEnum
from functools import lru_cache
import itertools
import os
from collections import namedtuple
from ctypes import POINTER
from dataclasses import dataclass

import pathlib
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import time
from typing import Any, Set, cast, Dict, List, NewType, Optional, Sequence, Tuple, Union

import geometry
import geometry as g
import gym
import math
import numpy as np
import pyglet
import yaml
from geometry import SE2value
from gym import spaces
from gym.utils import seeding
from numpy.random.mtrand import RandomState
from pyglet import gl, image, window

from duckietown_world import (
    get_DB18_nominal,
    get_DB18_uncalibrated,
    get_texture_file,
    MapFormat1,
    MapFormat1Constants,
    MapFormat1Constants as MF1C,
    MapFormat1Object,
    SE2Transform,
    DynamicModelParameters,
    DynamicModel,
)
from duckietown_world.world_duckietown.dynamics_delay import DelayedDynamics
from duckietown_world.gltf.export import get_duckiebot_color_from_colorname
from duckietown_world.resources import get_resource_path
from duckietown_world.world_duckietown.map_loading import get_transform
from gym_duckietown import list_maps2
from gym_duckietown.check_hw import get_graphics_information
from gym_duckietown.collision import (
    agent_boundbox,
    generate_norm,
    intersects,
    safety_circle_intersection,
    safety_circle_overlap,
    tile_corners,
)
from gym_duckietown.distortion import Distortion
from gym_duckietown.exceptions import InvalidMapException, NotInLane
from gym_duckietown.graphics import (
    photos_segmented_path,
    bezier_closest,
    bezier_draw,
    bezier_point,
    bezier_tangent,
    create_frame_buffers,
    gen_rot_matrix,
    load_texture,
    rotate_point,
    Texture,
)
from gym_duckietown.objects import (
    CheckerboardObj,
    DuckiebotObj,
    DuckieObj,
    TrafficLightObj,
    WorldObj,
)
from gym_duckietown.objmesh import get_mesh, MatInfo, ObjMesh
from gym_duckietown.randomization import Randomizer
from gym_duckietown.utils import get_subdir_path

from . import logger

DIM = 0.5

MAX_COMMAND = 1

MAX_WHEEL_ANG_VEL = 20
        

TileKind = NewType("TileKind", str)


class TileDict(TypedDict):
    # {"coords": (i, j), "kind": kind, "angle": angle, "drivable": drivable})
    coords: Tuple[int, int]
    kind: TileKind
    angle: int
    drivable: bool
    texture: Texture
    color: np.ndarray
    curves: Any


@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float


@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float


class MotionModelType(Enum):
    DYNAMICS = 0
    KINEMATICS = 1

    def __str__(self):
        return self.name.lower()


# Rendering window size
DEFAULT_WINDOW_WIDTH = 400
DEFAULT_WINDOW_HEIGHT = 300

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# np.array([0.15, 0.15, 0.15])
GREEN = (0.0, 1.0, 0.0)
# Ground/floor color


# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 19.15

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 75

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13 + 0.02

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12

# Safety radius multiplier
SAFETY_RAD_MULT = 1.8

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.20
# approx 2 tiles/second

DEFAULT_FRAMERATE = 30

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = "udem1"

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = -1000

MAX_SPAWN_ATTEMPTS = 5000

LanePosition0 = namedtuple("LanePosition", "dist dot_dir angle_deg angle_rad")


def squared_2d_dist(a, b):
    return (a[0] - b[0]) ** 2 + (a[2] - b[2]) ** 2


class LanePosition(LanePosition0):
    def as_json_dict(self):
        """Serialization-friendly format."""
        return dict(
            dist=self.dist,
            dot_dir=self.dot_dir,
            angle_deg=self.angle_deg,
            angle_rad=self.angle_rad,
        )


class Simulator(gym.Env):
    """
    Simple road simulator to test RL training.
    Draws a road with turns using OpenGL, and simulates
    basic differential-drive dynamics.
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "app"],
        "video.frames_per_second": 30,
    }

    cur_pos: np.ndarray
    cam_offset: np.ndarray
    road_tile_size: float
    grid_width: int
    grid_height: int
    step_count: int
    timestamp: float
    np_random: RandomState
    grid: List[TileDict]

    def __init__(
        self,
        map_name: str = DEFAULT_MAP_NAME,
        max_steps: int = DEFAULT_MAX_STEPS,
        draw_curve: bool = False,
        draw_bbox: bool = False,
        duckiebot_color: str = "red",
        domain_rand: bool = True,
        frame_rate: float = DEFAULT_FRAMERATE,
        frame_skip: bool = DEFAULT_FRAME_SKIP,
        camera_width: int = DEFAULT_CAMERA_WIDTH,
        camera_height: int = DEFAULT_CAMERA_HEIGHT,
        robot_speed: float = DEFAULT_ROBOT_SPEED,
        accept_start_angle_deg=DEFAULT_ACCEPT_START_ANGLE_DEG,
        full_transparency: bool = False,
        user_tile_start=None,
        seed: int = None,
        distortion: bool = False,
        dynamics_rand: bool = False,
        camera_rand: bool = False,
        randomize_maps_on_reset: bool = False,
        num_tris_distractors: int = 12,
        color_ground: Sequence[float] = (0.15, 0.15, 0.15),
        color_sky: Sequence[float] = BLUE_SKY,
        style: str = "photos",
        enable_leds: bool = False,
        frustum_filtering: bool = True,
        frustum_filtering_min_arccos_threshold: float = 0,
        depth_filtering_factor: float = 5,
        depth_texture_resize_factor: float = 3,
        distant_texture_resize_factor: float = 1 / 4,
        force_texture_resize_floor: bool = True,
        window_width: int = DEFAULT_WINDOW_WIDTH,
        window_height: int = DEFAULT_WINDOW_HEIGHT,
        skip_frame_buffer_copy: bool = False,
        motion_model: MotionModelType = MotionModelType.DYNAMICS,
        motion_model_delay: float = 0.0,
        motion_model_wheel_distance: float = 0.102,
        motion_model_wheel_radius_left: float = 0.0335,
        motion_model_wheel_radius_right: float = 0.0335,
        motion_model_encoder_resolution_rad: float = 1 / 135,
        init_pose: Optional[np.ndarray] = None,
    ):
        """

        :param map_name:
        :param max_steps:
        :param draw_curve:
        :param draw_bbox:
        :param domain_rand: If true, applies domain randomization
        :param frame_rate:
        :param frame_skip:
        :param camera_width:
        :param camera_height:
        :param robot_speed:
        :param accept_start_angle_deg:
        :param full_transparency:
        :param user_tile_start: If None, sample randomly. Otherwise (i,j). Overrides map start tile
        :param seed:
        :param distortion: If true, distorts the image with fish-eye approximation
        :param dynamics_rand: If true, perturbs the trim of the Duckiebot
        :param camera_rand: If true randomizes over camera miscalibration
        :param randomize_maps_on_reset: If true, randomizes the map on reset (Slows down training)
        :param style: String that represent which tiles will be loaded. One of ["photos", "photos-segmentation", "segmentation", "smooth", "synthetic"]
        :param enable_leds: Enables LEDs drawing.
        """
        self.enable_leds = enable_leds
        # information = get_graphics_information()
        # logger.info(
        #     f"Information about the graphics card:",
        #     pyglet_version=pyglet.version,
        #     information=information,
        #     nvidia_around=os.path.exists("/proc/driver/nvidia/version"),
        # )

        self.skip_frame_buffer_copy = skip_frame_buffer_copy

        self.motion_model = motion_model
        self.motion_model_delay = motion_model_delay
        self.motion_model_wheel_distance = motion_model_wheel_distance
        self.motion_model_wheel_radius_left = motion_model_wheel_radius_left
        self.motion_model_wheel_radius_right = motion_model_wheel_radius_right
        self.motion_model_encoder_resolution_rad = motion_model_encoder_resolution_rad
        
        self.init_pose = init_pose

        # debug
        self.init_total_render_time = [0, 0]
        self.perspective_time = [0, 0]
        self.tiles_total_render_time = [0, 0]

        # first initialize the RNG
        self.seed_value = seed
        self.seed(seed=self.seed_value)
        self.num_tris_distractors = num_tris_distractors
        self.color_ground = color_ground
        self.color_sky = list(color_sky)

        self.duckiebot_color = duckiebot_color

        # If true, then we publish all transparency information
        self.full_transparency = full_transparency

        # Map name, set in _load_map()
        self.map_name = None
        self.next_map_name = None

        # Full map file path, set in _load_map()
        self.map_file_path = None

        # The parsed content of the map_file
        self.map_data = None

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Flag to draw bounding boxes
        self.draw_bbox = draw_bbox

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand

        self.randomizer = Randomizer()

        # Frame rate to run at
        self.frame_rate = frame_rate
        self.delta_time = 1.0 / self.frame_rate

        # Number of frames to skip per action
        self.frame_skip = frame_skip

        # Produce graphical output
        self.graphics = True

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.robot_speed = robot_speed
        # We observe an RGB image with pixels in [0, 255]
        # Note: the pixels are in uint8 format because this is more compact
        # than float32 if sent over the network or stored in a dataset
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.camera_height, self.camera_width, 3),
            dtype=np.uint8,
        )

        self.reward_range = (-1000, 1000)

        self.frustum_filtering = frustum_filtering
        self.frustum_filtering_min_arccos_threshold = frustum_filtering_min_arccos_threshold
        self.depth_filtering_factor = depth_filtering_factor
        self.depth_texture_resize_factor = depth_texture_resize_factor
        self.distant_texture_resize_factor = distant_texture_resize_factor
        self.force_texture_resize_floor = force_texture_resize_floor

        # Window for displaying the environment to humans
        self.window = None

        self.window_width = window_width
        self.window_height = window_height

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial", font_size=14, x=5, y=self.window_height - 19
        )

        # Create a frame buffer object for the observation
        self.multi_fbo, self.final_fbo = create_frame_buffers(
            self.camera_width, self.camera_height, 4
        )

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(
            self.window_width, self.window_height, 4
        )

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(
            shape=(self.window_height, self.window_width, 3), dtype=np.uint8
        )

        # allowed angle in lane for starting position
        self.accept_start_angle_deg = accept_start_angle_deg

        # Load the map
        self._load_map(map_name)

        self.squared_depth_filtering_thresh = (
            self.depth_filtering_factor * self.road_tile_size
        ) ** 2
        self.squared_depth_texture_resizing_thresh = (
            self.depth_texture_resize_factor * self.road_tile_size
        ) ** 2

        # Distortion params, if so, load the library, only if not bbox mode
        self.distortion = distortion and not draw_bbox
        self.camera_rand = False
        if not draw_bbox and distortion:
            if distortion:
                self.camera_rand = camera_rand

                self.camera_model = Distortion(camera_rand=self.camera_rand)

        # Used by the UndistortWrapper, always initialized to False
        self.undistort = False

        # Dynamics randomization
        self.dynamics_rand = dynamics_rand

        # Start tile
        self.user_tile_start = user_tile_start

        self.style = style

        self.randomize_maps_on_reset = randomize_maps_on_reset

        if self.randomize_maps_on_reset:
            self.map_names = os.listdir(get_subdir_path("maps"))
            self.map_names = [
                _map
                for _map in self.map_names
                if not _map.startswith(("calibration", "regress"))
            ]
            self.map_names = [
                mapfile.replace(".yaml", "") for mapfile in self.map_names
            ]

        # Initialize the state
        self.reset()

        self.last_action = np.array([0, 0])
        self.wheelVels = np.array([0, 0])

    def _init_vlists(self):

        ns = 8
        assert ns >= 2

        # half_size = self.road_tile_size / 2
        TS = self.road_tile_size

        def get_point(u, v):
            pu = u / (ns - 1)
            pv = v / (ns - 1)
            x = -TS / 2 + pu * TS
            z = -TS / 2 + pv * TS
            tu = pu
            tv = 1 - pv
            return (x, 0.0, z), (tu, tv)

        vertices = []
        textures = []
        normals = []
        colors = []
        for i, j in itertools.product(range(ns - 1), range(ns - 1)):
            tl_p, tl_t = get_point(i, j)
            tr_p, tr_t = get_point(i + 1, j)
            br_p, br_t = get_point(i, j + 1)
            bl_p, bl_t = get_point(i + 1, j + 1)
            normal = [0.0, 1.0, 0.0]

            color = (255, 255, 255, 255)
            vertices.extend(tl_p)
            textures.extend(tl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(tr_p)
            textures.extend(tr_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(bl_p)
            textures.extend(bl_t)
            normals.extend(normal)
            colors.extend(color)

            vertices.extend(br_p)
            textures.extend(br_t)
            normals.extend(normal)
            colors.extend(color)

            #
            # normals.extend([0.0, 1.0, 0.0] * 4)

        # def get_quad_vertices(cx, cz, hs) -> Tuple[List[float], List[float], List[float]]:
        #     v = [
        #         -hs + cx,
        #         0.0,
        #         -hs + cz,
        #         #
        #         hs + cx,
        #         0.0,
        #         -hs + cz,
        #         #
        #         hs + cx,
        #         0.0,
        #         hs + cz,
        #         #
        #         -hs + cx,
        #         0.0,
        #         hs + cz,
        #     ]
        #     n = [0.0, 1.0, 0.0] * 4
        #     t = [0.0, 1.0,
        #          #
        #          1.0, 1.0,
        #          #
        #          1.0, 0.0,
        #          #
        #          0.0, 0.0]
        #     return v, n, t

        # Create the vertex list for our road quad
        # Note: the vertices are centered around the origin so we can easily
        # rotate the tiles about their center

        # verts = []
        # texCoords = []
        # normals = []
        #
        # v, n, t = get_quad_vertices(cx=0, cz=0, hs=half_size)
        # verts.extend(v)
        # normals.extend(n)
        # texCoords.extend(t)

        # verts = [
        #     -half_size,
        #     0.0,
        #     -half_size,
        #     #
        #     half_size,
        #     0.0,
        #     -half_size,
        #     #
        #     half_size,
        #     0.0,
        #     half_size,
        #     #
        #     -half_size,
        #     0.0,
        #     half_size,
        # ]
        # texCoords = [1.0, 0.0,
        #              0.0, 0.0,
        #              0.0, 1.0,
        #              1.0, 1.0]
        # Previous choice would reflect the texture
        # logger.info(nv=len(vertices), nt=len(textures), nn=len(normals), vertices=vertices,
        # textures=textures,
        #             normals=normals)
        total = len(vertices) // 3
        self.road_vlist = pyglet.graphics.vertex_list(
            total,
            ("v3f", vertices),
            ("t2f", textures),
            ("n3f", normals),
            ("c4B", colors),
        )
        # logger.info("done")
        # Create the vertex list for the ground quad
        verts = [
            -1,
            -0.8,
            1,
            #
            -1,
            -0.8,
            -1,
            #
            1,
            -0.8,
            -1,  #
            1,
            -0.8,
            1,
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ("v3f", verts))

    def reset(self, segment: bool = False):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0
        self.timestamp = 0.0

        # Robot's current speed
        self.speed = 0.0
        if self.next_map_name is not None:
            map_name = self.next_map_name
            logger.info(f"Map requested by service: {map_name}")
            self._load_map(map_name)
            self.next_map_name = None
        elif self.randomize_maps_on_reset:
            map_name = self.np_random.choice(self.map_names)
            logger.info(f"Random map chosen: {map_name}")
            self._load_map(map_name)

        self.randomization_settings = self.randomizer.randomize(rng=self.np_random)

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        if self.domain_rand:
            horz_mode = self.randomization_settings["horz_mode"]
            if horz_mode == 0:
                self.horizon_color = self._perturb(self.color_sky)
            elif horz_mode == 1:
                self.horizon_color = self._perturb(WALL_COLOR)
            elif horz_mode == 2:
                self.horizon_color = self._perturb([0.15, 0.15, 0.15], 0.4)
            elif horz_mode == 3:
                self.horizon_color = self._perturb([0.9, 0.9, 0.9], 0.4)
        else:
            self.horizon_color = self.color_sky

        # Setup some basic lighting with a far away sun
        if self.domain_rand:
            light_pos = self.randomization_settings["light_pos"]
        else:
            # light_pos = [-40, 200, 100, 0.0]

            light_pos = [0.0, 3.0, 0.0, 1.0]

        # DIM = 0.0
        ambient = np.array([0.50 * DIM, 0.50 * DIM, 0.50 * DIM, 1])
        ambient = self._perturb(ambient, 0.3)
        diffuse = np.array([0.70 * DIM, 0.70 * DIM, 0.70 * DIM, 1])
        diffuse = self._perturb(diffuse, 0.99)
        # specular = np.array([0.3, 0.3, 0.3, 1])
        specular = np.array([0.0, 0.0, 0.0, 1])

        # logger.info(light_pos=light_pos, ambient=ambient, diffuse=diffuse, specular=specular)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*light_pos))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))

        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(0.4))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.3))
        # gl.glLightfv(gl.GL_LIGHT0, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.1))

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)

        # Ground color
        self.ground_color = self._perturb(np.array(self.color_ground), 0.3)

        # Distance between the robot's wheels
        self.wheel_dist = self._perturb(WHEEL_DIST)

        # Set default values

        # Distance bewteen camera and ground
        self.cam_height = CAMERA_FLOOR_DIST

        # Angle at which the camera is rotated
        self.cam_angle = [CAMERA_ANGLE, 0, 0]

        # Field of view angle of the camera
        self.cam_fov_y = CAMERA_FOV_Y

        # Perturb using randomization API (either if domain rand or only camera rand
        if self.domain_rand or self.camera_rand:
            self.cam_height *= self.randomization_settings["camera_height"]
            self.cam_angle = [
                CAMERA_ANGLE * self.randomization_settings["camera_angle"],
                0,
                0,
            ]
            self.cam_fov_y *= self.randomization_settings["camera_fov_y"]

        # Camera offset for use in free camera mode
        self.cam_offset = np.array([0, 0, 0])

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = self.num_tris_distractors
        verts = []
        colors = []
        for _ in range(0, 3 * numTris):
            p = self.np_random.uniform(
                low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,)
            )
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]

        self.tri_vlist = pyglet.graphics.vertex_list(
            3 * numTris, ("v3f", verts), ("c3f", colors)
        )

        # Randomize tile parameters
        for tile in self.grid:
            rng = self.np_random if self.domain_rand else None

            kind = tile["kind"]
            if self.style == "photos-segmentation":
                fn = [f"{photos_segmented_path}/{kind}.jpg"][0]
            else:
                fn = get_texture_file(f"tiles-processed/{self.style}/{kind}/texture")[0]
            tile["texture"] = {}
            # ft = get_fancy_textures(self.style, texture_name)
            t = load_texture(fn, segment=False, segment_into_color=False)
            tt = Texture(t, tex_name=kind, rng=rng, style=self.style)
            tile["texture"]["original"] = tt
            t_resized = load_texture(
                fn,
                segment=False,
                segment_into_color=False,
                resize_factor=self.distant_texture_resize_factor,
            )
            tt_resized = Texture(t_resized, tex_name=kind, rng=rng, style=self.style)
            tile["texture"]["resized"] = tt_resized

            # Random tile color multiplier
            tile["color"] = self._perturb([1, 1, 1, 1], 0.2)

        # Randomize object parameters
        for obj in self.objects:
            # Randomize the object color
            obj.color = self._perturb([1, 1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            if obj.optional and self.domain_rand:
                obj.visible = self.np_random.integers(0, 2) == 0
            else:
                obj.visible = True
                
        self.tile_texture_group = {}
        self.texture_dict = {}
        self.texture_to_kind_dict = {}
        for i, j in itertools.product(range(self.grid_width), range(self.grid_height)):
            # Get the tile type and angle
            tile = self._get_tile(i, j)
            kind = tile["kind"]
            
            textures = tile["texture"]
            texture_name = textures["original"].tex_name
            
            if texture_name not in self.texture_dict:
                self.texture_dict[texture_name] = textures
                
            if texture_name not in self.texture_to_kind_dict:
                self.texture_to_kind_dict[texture_name] = kind
                
            if texture_name in self.tile_texture_group:
                self.tile_texture_group[texture_name].append((i, j))
            else:
                self.tile_texture_group[texture_name] = [(i, j)]
                
            
        # If the params specifies an initial pose
        if self.init_pose is not None:
            self.cur_pos = np.array([self.init_pose[0], 0, -self.init_pose[1]])
            self.cur_angle = self.init_pose[2]
            logger.info(f"Using initial pose: position: {[self.cur_pos[0], -self.cur_pos[2]]}, angle: {self.cur_angle}")
        else:
            # If the map specifies a starting tile
            if self.user_tile_start:
                logger.info(f"Using user tile start: {self.user_tile_start}")
                i, j = self.user_tile_start
                tile = self._get_tile(i, j)
                if tile is None:
                    msg = "The tile specified does not exist."
                    raise Exception(msg)
                logger.debug(f"tile: {tile}")
            else:
                if self.start_tile is not None:
                    tile = self.start_tile
                    logger.info(f"Using map tile start: {tile['coords']}")
                else:
                    # Select a random drivable tile to start on
                    if not self.drivable_tiles:
                        msg = "There are no drivable tiles. Use start_tile or self.user_tile_start"
                        raise Exception(msg)
                    tile_idx = self.np_random.integers(0, len(self.drivable_tiles))
                    tile = self.drivable_tiles[tile_idx]
                    logger.info(f"Randomly selected tile start: {tile['coords']}")

            # If the map specifies a starting pose
            if self.start_pose is not None:
                
                i, j = tile["coords"]
                x = i * self.road_tile_size + self.start_pose[0][0]
                z = j * self.road_tile_size + self.start_pose[0][2]
                propose_pos = np.array([x, 0, z])
                propose_angle = self.start_pose[1]

                logger.info(
                    f"Using map pose start. Position: {[propose_pos[0], -propose_pos[2]]}, Angle: {propose_angle}"
                    " (relative to tile start)"
                )

            else:
                # Keep trying to find a valid spawn position on this tile
                logger.info("Sampling random pose start")
                for _ in range(MAX_SPAWN_ATTEMPTS):
                    i, j = tile["coords"]

                    # Choose a random position on this tile
                    x = self.np_random.uniform(i, i + 1) * self.road_tile_size
                    z = self.np_random.uniform(j, j + 1) * self.road_tile_size
                    propose_pos = np.array([x, 0, z])

                    # Choose a random direction
                    propose_angle = self.np_random.uniform(0, 2 * math.pi)

                    # logger.debug('Sampled %s %s angle %s' % (propose_pos[0],
                    #                                          propose_pos[1],
                    #                                          np.rad2deg(propose_angle)))

                    # If this is too close to an object or not a valid pose, retry
                    inconvenient = self._inconvenient_spawn(propose_pos)

                    if inconvenient:
                        # msg = 'The spawn was inconvenient.'
                        # logger.warning(msg)
                        continue

                    invalid = not self._valid_pose(
                        propose_pos, propose_angle, safety_factor=1.3
                    )
                    if invalid:
                        # msg = 'The spawn was invalid.'
                        # logger.warning(msg)
                        continue

                    # If the angle is too far away from the driving direction, retry
                    try:
                        lp = self.get_lane_pos2(propose_pos, propose_angle)
                    except NotInLane:
                        continue
                    M = self.accept_start_angle_deg
                    ok = -M < lp.angle_deg < +M
                    if not ok:
                        continue
                    # Found a valid initial pose
                    logger.info(
                        f"Found a valid starting pose after {_} attempts: "
                        f"position {[propose_pos[0], -propose_pos[2]]}, angle {propose_angle}"
                        " (relative to tile start)"
                    )
                    break
                else:
                    
                    propose_pos = np.array([1, 0, 1])
                    propose_angle = 1
                    logger.warning(f"Could not find a valid starting pose after {MAX_SPAWN_ATTEMPTS} attempts,"
                                f" using hardcoded pose: position {[propose_pos[0], -propose_pos[2]]}, angle {propose_angle}"
                                " (relative to tile start)")

                    # raise Exceptipathlib

            self.cur_pos = propose_pos
            self.cur_angle = propose_angle

        init_vel = np.array([0, 0])

        # Initialize Dynamics model
        if self.dynamics_rand:
            trim = 0 + self.randomization_settings["trim"][0]
            p = get_DB18_uncalibrated(delay=self.motion_model_delay, trim=trim)
        else:
            p = get_DB18_nominal(delay=self.motion_model_delay)

        q = self.cartesian_from_weird(self.cur_pos, self.cur_angle)
        v0 = geometry.se2_from_linear_angular(init_vel, 0)
        c0 = q, v0
        self.state = p.initialize(c0=c0, t0=0)

        # modify the internal motor parameters of the dynamics model
        # to account for the wheel parameters
        if isinstance(self.state, DelayedDynamics):
            self.is_delay_dynamics = True
            parameters = self.state.state.parameters
        elif isinstance(self.state, DynamicModel):
            self.is_delay_dynamics = False
            parameters = self.state.parameters
        else:
            raise ValueError(f"Unexpected dynamic model {type(self.state)}")

        parameters.wheel_distance = (
            self.motion_model_wheel_distance
        )
        parameters.wheel_radius_right = (
            self.motion_model_wheel_radius_right
        )
        parameters.wheel_radius_left = (
            self.motion_model_wheel_radius_left
        )
        parameters.encoder_resolution_rad = (
            self.motion_model_encoder_resolution_rad
        )
        
        max_lin_vel = MAX_WHEEL_ANG_VEL * (parameters.wheel_radius_right + parameters.wheel_radius_left) / 2
        max_ang_vel = MAX_WHEEL_ANG_VEL * (parameters.wheel_radius_right + parameters.wheel_radius_left) / parameters.wheel_distance
        
        # dynamics formula:
        # current_lin_acc = (u_alpha_r_coeff * wheel_radius_r * command_r + u_alpha_l_coeff * wheel_radius_l * command_l) - u_coeff * prev_lin_vel
        # current_ang_acc = (w_alpha_r_coeff * wheel_radius_r * command_r - w_alpha_l_coeff * wheel_radius_l * command_l) - w_coeff * prev_ang_vel
        
        # these coefficients influence the dynamic response of the robot:
        # the higher the coefficients, the more aggressive the response to commands
        u_alpha_r_coeff = 45
        u_alpha_l_coeff = 45
        w_alpha_r_coeff = u_alpha_r_coeff / parameters.wheel_distance
        w_alpha_l_coeff = u_alpha_l_coeff / parameters.wheel_distance
        
        # these coefficients influence the steady-state velocity of the robot:
        # the higher the coefficients, the lower the steady-state velocity
        # Try to use the wheel parameters to calculate the coefficients!
        u_coeff = MAX_COMMAND * (u_alpha_r_coeff * parameters.wheel_radius_right + u_alpha_l_coeff * parameters.wheel_radius_left) / max_lin_vel
        w_coeff = MAX_COMMAND * (w_alpha_r_coeff * parameters.wheel_radius_right + w_alpha_l_coeff * parameters.wheel_radius_left) / max_ang_vel
        
        parameters.u_alpha_r = u_alpha_r_coeff * parameters.wheel_radius_right
        parameters.u_alpha_l = u_alpha_l_coeff * parameters.wheel_radius_left
        parameters.w_alpha_r = w_alpha_r_coeff * parameters.wheel_radius_right
        parameters.w_alpha_l = w_alpha_l_coeff * parameters.wheel_radius_left
        
        parameters.u1 = u_coeff
        parameters.u2 = 0
        parameters.u3 = 0
        parameters.w1 = w_coeff
        parameters.w2 = 0
        parameters.w3 = 0
        
        if self.is_delay_dynamics:
            self.state.state.parameters = parameters
        else:
            self.state.parameters = parameters

        logger.debug(f"Starting at {self.cur_pos} {self.cur_angle}")

        # Generate the first camera image
        obs = self.render_obs(segment=segment)

        # Return first observation
        return obs

    @staticmethod
    def get_set_available_maps():
        def get_set_custom_maps() -> Set[str]:
            # list all the files terminating in .yaml inside the maps folder
            maps_folder = pathlib.Path(__file__).parent.parent / "maps"
            if maps_folder.exists():
                maps = set(
                    [
                        f.stem
                        for f in maps_folder.iterdir()
                        if (f.is_file() and f.suffix == ".yaml")
                    ]
                )
            else:
                logger.warning(
                    f"maps folder {maps_folder} for custom maps does not exist"
                )
                maps = set()
            return maps

        return set(list_maps2().keys()).union(get_set_custom_maps())

    def _load_map(self, map_name: str):
        """
        Load the map layout from a YAML file
        """

        # Store the map name
        if os.path.exists(map_name) and os.path.isfile(map_name):
            # if env is loaded using gym's register function, we need to extract the map name from the complete url
            map_name = os.path.basename(map_name)
            assert map_name.endswith(".yaml")
            map_name = ".".join(map_name.split(".")[:-1])
        self.map_name = map_name

        # Get the full map file path
        try:
            self.map_file_path = self._get_custom_map_path(map_name)
            logger.debug(f'loading custom map file "{self.map_file_path}"')
        except FileNotFoundError:
            # if the map is not found in the custom maps folder, we search in duckietown_world library
            self.map_file_path = get_resource_path(f"{map_name}.yaml")
            logger.debug(f'loading map file "{self.map_file_path}"')

        with open(self.map_file_path, "r") as f:
            self.map_data = yaml.load(f, Loader=yaml.Loader)

        self._interpret_map(self.map_data)

    def _get_custom_map_path(self, map_name: str) -> str:
        custom_map_path_test = (
            pathlib.Path(__file__).parent.parent / f"maps/{map_name}.yaml"
        )
        if os.path.exists(custom_map_path_test) and os.path.isfile(
            custom_map_path_test
        ):
            return custom_map_path_test
        else:
            raise FileNotFoundError(
                f"Custom map {map_name} not found at {custom_map_path_test}"
            )

    def _interpret_map(self, map_data: MapFormat1):
        try:
            if not "tile_size" in map_data:
                msg = "Must now include explicit tile_size in the map data."
                raise InvalidMapException(msg)

            self.road_tile_size = map_data["tile_size"]
            if self.road_tile_size != 0.585:
                raise InvalidMapException(
                    "Only tile_size=0.585 is supported in the map data!"
                    f" Do you really need to use a different"
                    f" tile_size={self.road_tile_size}? Ask Nicola."
                )
            assert self.road_tile_size > 0
            self._init_vlists()

            tiles = map_data["tiles"]
            assert len(tiles) > 0
            assert len(tiles[0]) > 0

            # Create the grid
            self.grid_height = len(tiles)
            self.grid_width = len(tiles[0])
            # noinspection PyTypeChecker
            self.grid = [None] * self.grid_width * self.grid_height

            # We keep a separate list of drivable tiles
            self.drivable_tiles = []

            # For each row in the grid
            for j, row in enumerate(tiles):

                if len(row) != self.grid_width:
                    msg = "each row of tiles must have the same length"
                    raise InvalidMapException(msg, row=row)

                # For each tile in this row
                for i, tile in enumerate(row):
                    tile = tile.strip()

                    if tile == "empty":
                        continue

                    directions = ["S", "E", "N", "W"]
                    default_orient = "E"

                    if "/" in tile:
                        kind, orient = tile.split("/")
                        kind = kind.strip(" ")
                        orient = orient.strip(" ")
                        angle = directions.index(orient)

                    elif "4" in tile:
                        kind = "4way"
                        angle = directions.index(default_orient)

                    else:
                        kind = tile
                        angle = directions.index(default_orient)

                    DRIVABLE_TILES = [
                        "straight",
                        "curve_left",
                        "curve_right",
                        "3way_left",
                        "3way_right",
                        "4way",
                    ]
                    drivable = kind in DRIVABLE_TILES

                    # logger.info(f'kind {kind} drivable {drivable} row = {row}')

                    tile = cast(
                        TileDict,
                        {
                            "coords": (i, j),
                            "kind": kind,
                            "angle": angle,
                            "drivable": drivable,
                        },
                    )

                    self._set_tile(i, j, tile)

                    if drivable:
                        tile["curves"] = self._get_curve(i, j)
                        self.drivable_tiles.append(tile)

            self.mesh = get_duckiebot_mesh(self.duckiebot_color)
            self._load_objects(map_data)

            # Get the starting tile from the map, if specified
            self.start_tile = None
            if "start_tile" in map_data:
                coords = map_data["start_tile"]
                self.start_tile = self._get_tile(*coords)

            # Get the starting pose from the map, if specified
            self.start_pose = None
            if "start_pose" in map_data:
                self.start_pose = map_data["start_pose"]
        except Exception as e:
            msg = "Cannot load map data"
            raise InvalidMapException(msg, map_data=map_data)

    def _load_objects(self, map_data: MapFormat1):
        # Create the objects array
        self.objects = []

        # The corners for every object, regardless if collidable or not
        self.object_corners = []

        # Arrays for checking collisions with N static objects
        # (Dynamic objects done separately)
        # (N x 2): Object position used in calculating reward
        self.collidable_centers = []

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.collidable_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.collidable_norms = []

        # (N): Safety radius for object used in calculating reward
        self.collidable_safety_radii = []

        # For each object
        try:
            objects = map_data["objects"]
        except KeyError:
            pass
        else:
            if isinstance(objects, list):
                for obj_idx, desc in enumerate(objects):
                    kind = desc["kind"]
                    obj_name = f"ob{obj_idx:02d}-{kind}"
                    self.interpret_object(obj_name, desc)
            elif isinstance(objects, dict):
                for obj_name, desc in objects.items():
                    self.interpret_object(obj_name, desc)
            else:
                raise ValueError(objects)

        # If there are collidable objects
        if len(self.collidable_corners) > 0:
            self.collidable_corners = np.stack(self.collidable_corners, axis=0)
            self.collidable_norms = np.stack(self.collidable_norms, axis=0)

            # Stack doesn't do anything if there's only one object,
            # So we add an extra dimension to avoid shape errors later
            if len(self.collidable_corners.shape) == 2:
                self.collidable_corners = self.collidable_corners[np.newaxis]
                self.collidable_norms = self.collidable_norms[np.newaxis]

        self.collidable_centers = np.array(self.collidable_centers)
        self.collidable_safety_radii = np.array(self.collidable_safety_radii)

    def interpret_object(self, objname: str, desc: MapFormat1Object):
        kind = desc["kind"]

        W = self.grid_width
        tile_size = self.road_tile_size
        transform: SE2Transform = get_transform(desc, W, tile_size)
        # logger.info(desc=desc, transform=transform)

        pose = transform.as_SE2()

        pos, angle_rad = self.weird_from_cartesian(pose)

        # c = self.cartesian_from_weird(pos, angle_rad)
        # logger.debug(desc=desc, pose=geometry.SE2.friendly(pose), weird=(pos, angle_rad),
        # c=geometry.SE2.friendly(c))

        # pos = desc["pos"]
        # x, z = pos[0:2]
        # y = pos[2] if len(pos) == 3 else 0.0

        # rotate = desc.get("rotate", 0.0)
        optional = desc.get("optional", False)

        # pos = self.road_tile_size * np.array((x, y, z))

        # Load the mesh

        if kind == MapFormat1Constants.KIND_DUCKIEBOT:
            use_color = desc.get("color", self.duckiebot_color)

            mesh = get_duckiebot_mesh(use_color)

        elif kind.startswith("sign"):
            change_materials: Dict[str, MatInfo]
            # logger.info(kind=kind, desc=desc)
            minfo = cast(MatInfo, {"map_Kd": f"{kind}.png"})
            change_materials = {"April_Tag": minfo}
            mesh = get_mesh("sign_generic", change_materials=change_materials)
        elif kind == "floor_tag":
            return
        else:
            mesh = get_mesh(kind)

        if "height" in desc:
            scale = desc["height"] / mesh.max_coords[1]
        else:
            if "scale" in desc:
                scale = desc["scale"]
            else:
                scale = 1.0
        assert not (
            "height" in desc and "scale" in desc
        ), "cannot specify both height and scale"

        static = desc.get("static", True)
        # static = desc.get('static', False)
        # print('static is now', static)

        obj_desc = {
            "kind": kind,
            "mesh": mesh,
            "pos": pos,
            "angle": angle_rad,
            "scale": scale,
            "optional": optional,
            "static": static,
        }

        if static:
            if kind == MF1C.KIND_TRAFFICLIGHT:
                obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
            else:
                obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
        else:
            if kind == MF1C.KIND_DUCKIEBOT:
                obj = DuckiebotObj(
                    obj_desc,
                    self.domain_rand,
                    SAFETY_RAD_MULT,
                    WHEEL_DIST,
                    ROBOT_WIDTH,
                    ROBOT_LENGTH,
                )
            elif kind == MF1C.KIND_DUCKIE:
                obj = DuckieObj(
                    obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size
                )
            elif kind == MF1C.KIND_CHECKERBOARD:
                obj = CheckerboardObj(
                    obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size
                )
            else:
                msg = "Object kind unknown."
                raise InvalidMapException(msg, kind=kind)

        self.objects.append(obj)

        # Compute collision detection information

        # angle = rotate * (math.pi / 180)

        # # Find drivable tiles object could intersect with
        # # possible_tiles = find_candidate_tiles(obj.obj_corners, self.road_tile_size)

        # If the object intersects with a drivable tile
        if (
            static
            and kind != MF1C.KIND_TRAFFICLIGHT
            # We want collision checking also for things outside the lanes
            # # and self._collidable_object(obj.obj_corners, obj.obj_norm, possible_tiles)
        ):
            # noinspection PyUnresolvedReferences
            self.collidable_centers.append(
                pos
            )  # XXX: changes types during initialization
            self.collidable_corners.append(obj.obj_corners.T)
            self.collidable_norms.append(obj.obj_norm)
            # noinspection PyUnresolvedReferences
            self.collidable_safety_radii.append(
                obj.safety_radius
            )  # XXX: changes types during initialization

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _set_tile(self, i: int, j: int, tile: TileDict) -> None:
        assert 0 <= i < self.grid_width
        assert 0 <= j < self.grid_height
        index: int = j * self.grid_width + i
        self.grid[index] = tile

    def _get_tile(self, i: int, j: int) -> Optional[TileDict]:
        """
        Returns None if the duckiebot is not in a tile.
        """
        i = int(i)
        j = int(j)
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        return self.grid[j * self.grid_width + i]

    def _perturb(
        self, val: Union[float, np.ndarray, List[float]], scale: float = 0.1
    ) -> np.ndarray:
        """
        Add noise to a value. This is used for domain randomization.
        """
        assert 0 <= scale < 1

        val = np.array(val)

        if not self.domain_rand:
            return val

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(
                low=1 - scale, high=1 + scale, size=val.shape
            )
            if val.size == 4:
                noise[3] = 1
        else:
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale)

        res = val * noise

        return res

    def _collidable_object(self, obj_corners, obj_norm, possible_tiles):
        """
        A function to check if an object intersects with any
        drivable tiles, which would mean our agent could run into them.
        Helps optimize collision checking with agent during runtime
        """

        if possible_tiles.shape == 0:
            return False

        drivable_tiles = []
        for c in possible_tiles:
            tile = self._get_tile(c[0], c[1])
            if tile and tile["drivable"]:
                drivable_tiles.append((c[0], c[1]))

        if not drivable_tiles:
            return False

        drivable_tiles = np.array(drivable_tiles)

        # Tiles are axis aligned, so add normal vectors in bulk
        tile_norms = np.array([[1, 0], [0, 1]] * len(drivable_tiles))

        # None of the candidate tiles are drivable, don't add object
        if len(drivable_tiles) == 0:
            return False

        # Find the corners for each candidate tile
        drivable_tiles = np.array(
            [
                tile_corners(
                    self._get_tile(pt[0], pt[1])["coords"], self.road_tile_size
                ).T
                for pt in drivable_tiles
            ]
        )

        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        if len(tile_norms.shape) == 2:
            tile_norms = tile_norms[np.newaxis]
        else:  # Stack works as expected
            drivable_tiles = np.stack(drivable_tiles, axis=0)
            tile_norms = np.stack(tile_norms, axis=0)

        # Only add it if one of the vertices is on a drivable tile
        return intersects(obj_corners, drivable_tiles, obj_norm, tile_norms)

    def get_grid_coords(self, abs_pos: np.array) -> Tuple[int, int]:
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position

        x-axis maps to increasing i indices
        z-axis maps to increasing j indices

        Note: may return coordinates outside of the grid if the
        position entered is outside of the grid.
        """

        x, _, z = abs_pos
        i = math.floor(x / self.road_tile_size)
        j = math.floor(z / self.road_tile_size)

        return int(i), int(j)

    def _get_curve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """
        tile = self._get_tile(i, j)
        assert tile is not None

        kind = tile["kind"]
        angle = tile["angle"]

        # Each tile will have a unique set of control points,
        # Corresponding to each of its possible turns

        if kind.startswith("straight"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_left":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.30],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        elif kind == "curve_right":
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                        [
                            [-0.50, 0, 0.20],
                            [-0.30, 0, 0.20],
                            [0.30, 0, 0.00],
                            [0.20, 0, -0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Hardcoded all curves for 3way intersection
        elif kind.startswith("3way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.25],
                            [0.20, 0, -0.25],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [0.20, 0, -0.20],
                            [0.20, 0, -0.50],
                        ],
                        [
                            [0.20, 0, 0.50],
                            [0.20, 0, 0.20],
                            [0.30, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [0.50, 0, -0.20],
                            [0.30, 0, -0.20],
                            [-0.20, 0, 0.00],
                            [-0.20, 0, 0.50],
                        ],
                    ]
                )
                * self.road_tile_size
            )

        # Template for each side of 4way intersection
        elif kind.startswith("4way"):
            pts = (
                np.array(
                    [
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, 0.00],
                            [0.00, 0, 0.20],
                            [0.50, 0, 0.20],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.25],
                            [-0.20, 0, 0.25],
                            [-0.20, 0, 0.50],
                        ],
                        [
                            [-0.20, 0, -0.50],
                            [-0.20, 0, -0.20],
                            [-0.30, 0, -0.20],
                            [-0.50, 0, -0.20],
                        ],
                    ]
                )
                * self.road_tile_size
            )
        else:
            msg = "Cannot get bezier for kind"
            raise InvalidMapException(msg, kind=kind)

        # Rotate and align each curve with its place in global frame
        if kind.startswith("4way"):
            fourway_pts = []
            # Generate all four sides' curves,
            # with 3-points template above
            for rot in np.arange(0, 4):
                mat = gen_rot_matrix(np.array([0, 1, 0]), rot * math.pi / 2)
                pts_new = np.matmul(pts, mat)
                pts_new += np.array(
                    [
                        (i + 0.5) * self.road_tile_size,
                        0,
                        (j + 0.5) * self.road_tile_size,
                    ]
                )
                fourway_pts.append(pts_new)

            fourway_pts = np.reshape(np.array(fourway_pts), (12, 4, 3))
            return fourway_pts

        # Hardcoded each curve; just rotate and shift
        elif kind.startswith("3way"):
            threeway_pts = []
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts_new = np.matmul(pts, mat)
            pts_new += np.array(
                [(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size]
            )
            threeway_pts.append(pts_new)

            threeway_pts = np.array(threeway_pts)
            threeway_pts = np.reshape(threeway_pts, (6, 4, 3))
            return threeway_pts

        else:
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts = np.matmul(pts, mat)
            pts += np.array(
                [(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size]
            )

        return pts

    def closest_curve_point(
        self, pos: np.array, angle: float
    ) -> Tuple[Optional[np.array], Optional[np.array]]:
        """
        Get the closest point on the curve to a given point
        Also returns the tangent at that point.

        Returns None, None if not in a lane.
        """

        i, j = self.get_grid_coords(pos)
        tile = self._get_tile(i, j)

        if tile is None or not tile["drivable"]:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = self._get_tile(i, j)["curves"]
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dir_vec = get_dir_vec(angle)

        dot_prods = np.dot(curve_headings, dir_vec)

        # Closest curve = one with largest dotprod
        cps = curves[np.argmax(dot_prods)]

        # Find closest point and tangent to this curve
        t = bezier_closest(cps, pos)
        point = bezier_point(cps, t)
        tangent = bezier_tangent(cps, t)

        return point, tangent

    def get_lane_pos2(self, pos, angle):
        """
        Get the position of the agent relative to the center of the right lane

        Raises NotInLane if the Duckiebot is not in a lane.
        """

        # Get the closest point along the right lane's Bezier curve,
        # and the tangent at that point
        point, tangent = self.closest_curve_point(pos, angle)
        if point is None or tangent is None:
            msg = f"Point not in lane: {pos}"
            raise NotInLane(msg)

        assert point is not None and tangent is not None

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = get_dir_vec(angle)
        dotDir = np.dot(dirVec, tangent)
        dotDir = np.clip(dotDir, -1.0, +1.0)

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = pos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle_rad = math.acos(dotDir)

        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1

        angle_deg = np.rad2deg(angle_rad)
        # return signedDist, dotDir, angle_deg

        return LanePosition(
            dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg, angle_rad=angle_rad
        )

    def _drivable_pos(self, pos) -> bool:
        """
        Check that the given (x,y,z) position is on a drivable tile
        """

        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        if tile is None:
            msg = f"No tile found at {pos} {coords}"
            logger.debug(msg)
            return False

        if not tile["drivable"]:
            msg = f"{pos} corresponds to tile at {coords} which is not drivable: {tile}"
            logger.debug(msg)
            return False

        return True

    def proximity_penalty2(self, pos: g.T3value, angle: float) -> float:
        """
        Calculates a 'safe driving penalty' (used as negative rew.)
        as described in Issue #24

        Describes the amount of overlap between the "safety circles" (circles
        that extend further out than BBoxes, giving an earlier collision 'signal'
        The number is max(0, prox.penalty), where a lower (more negative) penalty
        means that more of the circles are overlapping
        """

        pos = _actual_center(pos, angle)
        if len(self.collidable_centers) == 0:
            static_dist = 0

        # Find safety penalty w.r.t static obstacles
        else:
            d = np.linalg.norm(self.collidable_centers - pos, axis=1)

            if not safety_circle_intersection(
                d, AGENT_SAFETY_RAD, self.collidable_safety_radii
            ):
                static_dist = 0.0
            else:
                static_dist = safety_circle_overlap(
                    d, AGENT_SAFETY_RAD, self.collidable_safety_radii
                )

        total_safety_pen = static_dist
        for obj in self.objects:
            # Find safety penalty w.r.t dynamic obstacles
            total_safety_pen += obj.proximity(pos, AGENT_SAFETY_RAD)

        return total_safety_pen

    def _inconvenient_spawn(self, pos):
        """
        Check that agent spawn is not too close to any object
        """

        results = [
            np.linalg.norm(x.pos - pos)
            < max(x.max_coords) * 0.5 * x.scale + MIN_SPAWN_OBJ_DIST
            for x in self.objects
            if x.visible
        ]
        return np.any(results)

    def _collision(self, agent_corners):
        """
        Tensor-based OBB Collision detection
        """
        # Generate the norms corresponding to each face of BB
        agent_norm = generate_norm(agent_corners)

        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(
                agent_corners,
                self.collidable_corners,
                agent_norm,
                self.collidable_norms,
            )
            if collision:
                return True

        # Check collisions with Dynamic Objects
        for obj in self.objects:
            if obj.check_collision(agent_corners, agent_norm):
                return True

        # No collision with any object
        return False

    def _valid_pose(
        self, pos: g.T3value, angle: float, safety_factor: float = 1.0
    ) -> bool:
        """
        Check that the agent is in a valid pose

        safety_factor = minimum distance
        """

        # Compute the coordinates of the base of both wheels
        pos = _actual_center(pos, angle)
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

        # Recompute the bounding boxes (BB) for the agent
        agent_corners = get_agent_corners(pos, angle)
        no_collision = not self._collision(agent_corners)

        res = no_collision and all_drivable

        if not res:
            logger.debug(
                f"Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}"
            )
            logger.debug(f"safety_factor: {safety_factor}")
            logger.debug(f"pos: {pos}")
            logger.debug(f"l_pos: {l_pos}")
            logger.debug(f"r_pos: {r_pos}")
            logger.debug(f"f_pos: {f_pos}")

        return res

    def _check_intersection_static_obstacles(
        self, pos: g.T3value, angle: float
    ) -> bool:
        agent_corners = get_agent_corners(pos, angle)
        agent_norm = generate_norm(agent_corners)
        # logger.debug(agent_corners=agent_corners, agent_norm=agent_norm)
        # Check collisions with Static Objects
        if len(self.collidable_corners) > 0:
            collision = intersects(
                agent_corners,
                self.collidable_corners,
                agent_norm,
                self.collidable_norms,
            )
            if collision:
                return True
        return False

    cur_pose: np.ndarray
    cur_angle: float
    speed: float

    def update_physics(self, action, delta_time: Optional[float] = None):
        # print("updating physics")
        if delta_time is None:
            delta_time = self.delta_time
        self.wheelVels = action * self.robot_speed * 1
        prev_pos = self.cur_pos

        # Update the robot's position
        self.cur_pos, self.cur_angle = self._update_pos(action, delta_time)

        self.step_count += 1
        self.timestamp += delta_time

        self.last_action = action

        # Compute the robot's speed
        delta_pos = self.cur_pos - prev_pos
        self.speed = np.linalg.norm(delta_pos) / delta_time

        # Update world objects
        for obj in self.objects:
            if obj.kind == MapFormat1Constants.KIND_DUCKIEBOT:
                if not obj.static:
                    obj_i, obj_j = self.get_grid_coords(obj.pos)
                    same_tile_obj = [
                        o
                        for o in self.objects
                        if tuple(self.get_grid_coords(o.pos)) == (obj_i, obj_j)
                        and o != obj
                    ]

                    obj.step_duckiebot(
                        delta_time, self.closest_curve_point, same_tile_obj
                    )
            else:
                # print("stepping all objects")
                obj.step(delta_time)

    def get_agent_info(self) -> dict:
        info = {}
        pos = self.cur_pos
        angle = self.cur_angle
        # Get the position relative to the right lane tangent

        info["action"] = list(self.last_action)
        if self.full_transparency:
            #             info['desc'] = """
            #
            # cur_pos, cur_angle ::  simulator frame (non cartesian)
            #
            # egovehicle_pose_cartesian :: cartesian frame
            #
            #     the map goes from (0,0) to (grid_height, grid_width)*self.road_tile_size
            #
            # """
            try:
                lp = self.get_lane_pos2(pos, angle)
                info["lane_position"] = lp
            except NotInLane:
                info["lane_position"] = None

            info["robot_speed"] = self.speed
            info["proximity_penalty"] = self.proximity_penalty2(pos, angle)
            info["cur_pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
            info["cur_angle"] = float(angle)
            info["wheel_velocities"] = [self.wheelVels[0], self.wheelVels[1]]

            # put in cartesian coordinates
            # (0,0 is bottom left)
            # q = self.cartesian_from_weird(self.cur_pos, self.)
            # info['cur_pos_cartesian'] = [float(p[0]), float(p[1])]
            # info['egovehicle_pose_cartesian'] = {'~SE2Transform': {'p': [float(p[0]), float(p[1])],
            #                                                        'theta': angle}}

            info["timestamp"] = self.timestamp
            info["tile_coords"] = list(self.get_grid_coords(pos))
            # info['map_data'] = self.map_data
        misc = {}
        misc["Simulator"] = info
        return misc

    def cartesian_from_weird(self, pos, angle) -> np.ndarray:
        gx, gy, gz = pos
        grid_height = self.grid_height
        tile_size = self.road_tile_size

        # this was before but obviously doesn't work for grid_height = 1
        # cp = [gx, (grid_height - 1) * tile_size - gz]
        cp = [gx, grid_height * tile_size - gz]

        return geometry.SE2_from_translation_angle(np.array(cp), angle)

    def weird_from_cartesian(self, q: SE2value) -> Tuple[list, float]:

        cp, angle = geometry.translation_angle_from_SE2(q)

        gx = cp[0]
        gy = 0
        # cp[1] = (grid_height - 1) * tile_size - gz
        GH = self.grid_height
        tile_size = self.road_tile_size
        # this was before but obviously doesn't work for grid_height = 1
        # gz = (grid_height - 1) * tile_size - cp[1]
        gz = GH * tile_size - cp[1]
        return [gx, gy, gz], angle

    def compute_reward(self, pos, angle, speed):
        # Compute the collision avoidance penalty
        col_penalty = self.proximity_penalty2(pos, angle)

        # Get the position relative to the right lane tangent
        try:
            lp = self.get_lane_pos2(pos, angle)
        except NotInLane:
            reward = 40 * col_penalty
        else:

            # Compute the reward
            reward = (
                +1.0 * speed * lp.dot_dir + -10 * np.abs(lp.dist) + +40 * col_penalty
            )
        return reward

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        for _ in range(self.frame_skip):
            self.update_physics(action)

        # Generate the current camera image
        obs = self.render_obs()
        misc = self.get_agent_info()

        d = self._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc

    def _compute_done_reward(self) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose(self.cur_pos, self.cur_angle):
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
        else:
            done = False
            reward = self.compute_reward(self.cur_pos, self.cur_angle, self.robot_speed)
            msg = ""
            done_code = "in-progress"
        return DoneRewardInfo(
            done=done, done_why=msg, reward=reward, done_code=done_code
        )

    # def _is_world_point_projected_on_image(self, world_coords):
    #     world_coords = np.append(world_coords, 1)
    #     image_coords_tmp = np.dot(camera_intrinsics, np.dot(camera_extrinsics, world_coords))
    #     image_coords = image_coords_tmp / image_coords_tmp[2]
    #     if image_coords[2] < 0:
    #         return False
    #     return 0 <= image_coords[0] <= DEFAULT_CAMERA_WIDTH and 0 <= image_coords[1] <= DEFAULT_CAMERA_HEIGHT

    def _render_img(
        self,
        width: int,
        height: int,
        multi_fbo,
        final_fbo,
        img_array,
        top_down: bool = True,
        segment: bool = False,
    ) -> np.ndarray:
        """
        Render an image of the environment into a frame buffer
        Produce a numpy RGB array image as output
        """

        if not self.graphics:
            return np.zeros((height, width, 3), np.uint8)

        start_init_timer = time.perf_counter()

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        # pyglet.gl._shadow_window.switch_to()
        # TODO: this line takes 10% of rendering time
        self.shadow_window.switch_to()

        if segment:
            gl.glDisable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHTING)
            gl.glDisable(gl.GL_COLOR_MATERIAL)
        else:
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHTING)
            gl.glEnable(gl.GL_COLOR_MATERIAL)

        # note by default the ambient light is 0.2,0.2,0.2
        # ambient = [0.03, 0.03, 0.03, 1.0]
        ambient = [0.3, 0.3, 0.3, 1.0]

        gl.glEnable(gl.GL_POLYGON_SMOOTH)

        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        # Bind the multisampled frame buffer
        gl.glEnable(gl.GL_MULTISAMPLE)
        if self.skip_frame_buffer_copy:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
        gl.glViewport(0, 0, width, height)

        # Clear the color and depth buffers

        c0, c1, c2 = self.horizon_color if not segment else [255, 0, 255]
        gl.glClearColor(c0, c1, c2, 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(
            self.cam_fov_y,
            width / float(height),
            0.04,
            (
                max(0.05, self.depth_filtering_factor * self.road_tile_size)
                if (self.depth_filtering_factor >= 0 and not top_down)
                else 100.0
            ),
        )

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        pos = self.cur_pos
        angle = self.cur_angle
        # logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
        if self.domain_rand:
            pos = pos + self.randomization_settings["camera_noise"]

        x, y, z = pos + self.cam_offset
        dir_vec = get_dir_vec(angle)
        dx, dy, dz = dir_vec
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            gl.glRotatef(90, 1, 0, 0)
        elif not top_down:
            y += self.cam_height
            gl.glRotatef(self.cam_angle[0], 1, 0, 0)
            gl.glRotatef(self.cam_angle[1], 0, 1, 0)
            gl.glRotatef(self.cam_angle[2], 0, 0, 1)
            gl.glTranslatef(0, 0, CAMERA_FORWARD_DIST)

        if top_down:
            a = (self.grid_width * self.road_tile_size) / 2
            b = (self.grid_height * self.road_tile_size) / 2
            fov_y_deg = self.cam_fov_y
            fov_y_rad = np.deg2rad(fov_y_deg)
            H_to_fit = max(a, b) + 0.1  # borders

            H_FROM_FLOOR = H_to_fit / (np.tan(fov_y_rad / 2))

            look_from = a, H_FROM_FLOOR, b
            look_at = a, 0.0, b - 0.01
            up_vector = 0.0, 1.0, 0
            gl.gluLookAt(*look_from, *look_at, *up_vector)
        else:
            look_from = x, y, z
            look_at = x + dx, y + dy, z + dz
            up_vector = 0.0, 1.0, 0.0
            gl.gluLookAt(*look_from, *look_at, *up_vector)

        # Draw the ground quad
        gl.glDisable(gl.GL_TEXTURE_2D)
        # background is magenta when segmenting for easy isolation of main map image
        gl.glColor3f(*self.ground_color if not segment else [255, 0, 255])  # XXX
        gl.glPushMatrix()
        gl.glScalef(50, 0.01, 50)
        self.ground_vlist.draw(gl.GL_QUADS)
        gl.glPopMatrix()

        # Draw the ground/noise triangles
        if not segment:
            gl.glPushMatrix()
            gl.glTranslatef(0.0, 0.1, 0.0)
            self.tri_vlist.draw(gl.GL_TRIANGLES)
            gl.glPopMatrix()

        # Draw the road quads
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        add_lights = False
        if add_lights:
            for i in range(1):
                li = gl.GL_LIGHT0 + 1 + i
                # li_pos = [i + 1, 1, i + 1, 1]

                li_pos = [0.0, 0.2, 3.0, 1.0]
                diffuse = [0.0, 0.0, 1.0, 1.0] if i % 2 == 0 else [1.0, 0.0, 0.0, 1.0]
                ambient = [0.0, 0.0, 0.0, 1.0]
                specular = [0.0, 0.0, 0.0, 1.0]
                spot_direction = [0.0, -1.0, 0.0]
                logger.debug(
                    li=li,
                    li_pos=li_pos,
                    ambient=ambient,
                    diffuse=diffuse,
                    spot_direction=spot_direction,
                )
                gl.glLightfv(li, gl.GL_POSITION, (gl.GLfloat * 4)(*li_pos))
                gl.glLightfv(li, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
                gl.glLightfv(li, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*diffuse))
                gl.glLightfv(li, gl.GL_SPECULAR, (gl.GLfloat * 4)(*specular))
                gl.glLightfv(
                    li, gl.GL_SPOT_DIRECTION, (gl.GLfloat * 3)(*spot_direction)
                )
                # gl.glLightfv(li, gl.GL_SPOT_EXPONENT, (gl.GLfloat * 1)(64.0))
                gl.glLightf(li, gl.GL_SPOT_CUTOFF, 60)

                gl.glLightfv(li, gl.GL_CONSTANT_ATTENUATION, (gl.GLfloat * 1)(1.0))
                # gl.glLightfv(li, gl.GL_LINEAR_ATTENUATION, (gl.GLfloat * 1)(0.1))
                gl.glLightfv(li, gl.GL_QUADRATIC_ATTENUATION, (gl.GLfloat * 1)(0.2))
                gl.glEnable(li)

        # batch_road = pyglet.graphics.Batch()
        # For each grid tile

        end_init_timer = time.perf_counter()
        self.init_total_render_time[0] += end_init_timer - start_init_timer
        self.init_total_render_time[1] += 1
        print_render_time_every_k_steps = 200
        if self.init_total_render_time[1] % print_render_time_every_k_steps == 0:
            logger.debug_timing(
                f"Mean init render time last {print_render_time_every_k_steps} steps: "
                f" {1000*(self.init_total_render_time[0]/self.init_total_render_time[1]):.2f} ms"
            )
            self.init_total_render_time = [0, 0]

        start_rend_tiles_timer = time.perf_counter()
        
        if not self.domain_rand:
            gl.glColor4f(*[1,1,1,1])             
            
        for texture_name, tile_coords in self.tile_texture_group.items():
            
            # Bind the appropriate texture
            kind = self.texture_to_kind_dict[texture_name]
            if (self.force_texture_resize_floor and kind == "floor"):
                texture = self.texture_dict[texture_name]["resized"]
            else:
                texture = self.texture_dict[texture_name]["original"]
            texture.bind(segment)

            for i, j in tile_coords:
                # Get the tile type and angle
                tile = self._get_tile(i, j)

                if tile is None:
                    continue

                tile_xyz = np.array(
                    [(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size]
                )
                if self.frustum_filtering:
                    # filter out tiles that are outside camera frustum (use dot product of vector from camera to tile and camera direction)
                    if np.linalg.norm(tile_xyz-pos) > self.road_tile_size: # avoid filtering out tiles that are too close to the camera
                        tile_vec = tile_xyz - (pos - dir_vec * 0.1 * self.road_tile_size)
                        if (
                            np.dot(
                                dir_vec, tile_vec
                            ) / (np.linalg.norm(dir_vec) * np.linalg.norm(tile_vec))
                            < self.frustum_filtering_min_arccos_threshold
                        ):
                            continue

                if self.depth_filtering_factor >= 0:
                    # filter out tiles that are too far away from camera
                    if squared_2d_dist(tile_xyz, pos) > self.squared_depth_filtering_thresh:
                        continue

                # kind = tile["kind"]
                angle = tile["angle"]
                color = tile["color"]
                # if (self.force_texture_resize_floor and kind == "floor") or (
                #     self.depth_texture_resize_factor >= 0
                #     and squared_2d_dist(tile_xyz, pos)
                #     > self.squared_depth_texture_resizing_thresh
                # ):
                #     texture = self.texture_dict[texture_name]["resized"]
                # else:
                #     texture = self.texture_dict[texture_name]["original"]
                

                # # logger.info('drawing', tile_color=color)
                # TS = self.road_tile_size
                # vertices = [(i+0.5)*TS, 0, (j+0.5)*TS, (i-0.5)*TS, 0, (j+0.5)*TS, (i-0.5)*TS, 0, (j-0.5)*TS, (i+0.5)*TS, 0, (j-0.5)*TS]
                # texture_vertices = [1, 1, 0, 1, 0, 0, 1, 0]
                # vlist = batch_road.add(4, gl.GL_QUADS, None, ('v3f', vertices), ('t2f', texture_vertices))
                # gl.glBindTexture(texture.tex.target, texture.tex.id)

                if self.domain_rand:
                    gl.glColor4f(*color)
                gl.glPushMatrix()
                TS = self.road_tile_size
                gl.glTranslatef((i + 0.5) * TS, 0, (j + 0.5) * TS)
                gl.glRotatef(angle * 90 + 180, 0, 1, 0)

                # gl.glEnable(gl.GL_BLEND)
                # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

                self.road_vlist.draw(gl.GL_QUADS)
                # gl.glDisable(gl.GL_BLEND)

                gl.glPopMatrix()


        end_rend_tiles_timer = time.perf_counter()
        self.tiles_total_render_time[0] += end_rend_tiles_timer - start_rend_tiles_timer
        self.tiles_total_render_time[1] += 1
        print_render_time_every_k_steps = 200
        if self.tiles_total_render_time[1] % print_render_time_every_k_steps == 0:
            logger.debug_timing(
                f"Mean tiles render time last {print_render_time_every_k_steps} steps: "
                f" {1000*(self.tiles_total_render_time[0]/self.tiles_total_render_time[1]):.2f} ms"
            )
            self.tiles_total_render_time = [0, 0]

        # For each object
        for obj in self.objects:
            if self.frustum_filtering:
                # filter out objects that are outside the camera frustum (use dot product of vector from camera to object and camera direction)
                if np.linalg.norm(obj.pos-pos) > self.road_tile_size: # avoid filtering out objects that are too close to the camera
                    obj_vec = obj.pos - (pos - dir_vec * 0.1 * self.road_tile_size)
                    if (
                        np.dot(
                            dir_vec, obj_vec
                        ) / (np.linalg.norm(dir_vec) * np.linalg.norm(obj_vec))
                        < self.frustum_filtering_min_arccos_threshold
                    ):
                        continue
                    
            if self.depth_filtering_factor >= 0:
                # filter out objects that are too far away from camera
                if squared_2d_dist(obj.pos, pos) > self.squared_depth_filtering_thresh:
                    continue
            obj.render(
                draw_bbox=self.draw_bbox, segment=segment, enable_leds=self.enable_leds
            )

        # Draw the agent's own bounding box
        if self.draw_bbox:
            corners = get_agent_corners(pos, angle)
            gl.glColor3f(1, 0, 0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
            gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
            gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
            gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
            gl.glEnd()

        if top_down:
            gl.glPushMatrix()
            gl.glTranslatef(*self.cur_pos)
            gl.glScalef(1, 1, 1)
            gl.glRotatef(self.cur_angle * 180 / np.pi, 0, 1, 0)
            # glColor3f(*self.color)
            self.mesh.render()
            gl.glPopMatrix()
        draw_xyz_axes = False
        if draw_xyz_axes:
            draw_axes()
        # Resolve the multisampled frame buffer into the final frame buffer
        if not self.skip_frame_buffer_copy:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
            gl.glBlitFramebuffer(
                0,
                0,
                width,
                height,
                0,
                0,
                width,
                height,
                gl.GL_COLOR_BUFFER_BIT,
                gl.GL_LINEAR,
            )

            # Copy the frame buffer contents into a numpy array
            # Note: glReadPixels reads starting from the lower left corner
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        # TODO: this line takes 20% of rendering time
        gl.glReadPixels(
            0,
            0,
            width,
            height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            img_array.ctypes.data_as(POINTER(gl.GLubyte)),
        )

        # Unbind the frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def render_obs(self, segment: bool = False) -> np.ndarray:
        """
        Render an observation from the point of view of the agent
        """

        observation = self._render_img(
            self.camera_width,
            self.camera_height,
            self.multi_fbo,
            self.final_fbo,
            self.img_array,
            top_down=False,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort:
            observation = self.camera_model.distort(observation)

        return observation

    def render(self, mode: str = "human", close: bool = False, segment: bool = False):
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
        img = self._render_img(
            self.window_width,
            self.window_height,
            self.multi_fbo_human,
            self.final_fbo_human,
            self.img_array_human,
            top_down=top_down,
            segment=segment,
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort and mode != "free_cam":
            img = self.camera_model.distort(img)

        if mode == "rgb_array":
            return img

        if self.window is None:
            config = gl.Config(double_buffer=False)
            self.window = window.Window(
                width=self.window_width,
                height=self.window_height,
                resizable=False,
                config=config,
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Setup orghogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.window_width, 0, self.window_height, 0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = image.ImageData(
            width,
            height,
            "RGB",
            img.ctypes.data_as(POINTER(gl.GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(0, 0, 0, width=self.window_width, height=self.window_height)

        # Display position/state information
        if mode != "free_cam":
            x, y, z = self.cur_pos
            self.text_label.text = (
                f"pos: ({x:.2f}, {y:.2f}, {z:.2f}), angle: "
                f"{np.rad2deg(self.cur_angle):.1f} deg, steps: {self.step_count}, "
                f"speed: {self.speed:.2f} m/s"
            )
            self.text_label.draw()

        # Force execution of queued commands
        gl.glFlush()

        return img

    def set_motion_model(self, motion_model: MotionModelType):
        if not isinstance(motion_model, MotionModelType):
            raise ValueError(
                f"motion_model {motion_model} must be an instance of MotionModelType: "
                f"{MotionModelType.__members__.keys()}"
            )
        self.motion_model = motion_model

    def get_motion_model(self) -> MotionModelType:
        return self.motion_model

    def set_next_map_name(self, map_name: str):
        self.next_map_name = map_name

    def get_next_map_name(self) -> str:
        return self.next_map_name
    
    @lru_cache(maxsize=10)
    def get_kinematics_params_from_dynamics_params(self, dynamics_params: DynamicModelParameters, delta_time: float) -> DynamicModelParameters:
        """
        Mimic the kinematic parameters from the dynamic parameters
        """
        
        # Compute dynamics parameters that mimic a kinematic model,
        # i.e. a immediate reaction in changing velocities (reach steady-state velocity in one time step),
        # paying attention to achieve the same target speed as with the dynamics model
        kinematics_param_lin = (
            (
                (dynamics_params.u_alpha_r + dynamics_params.u_alpha_l)
                / dynamics_params.u1
            )
            / delta_time
        ) / 2
        kinematics_param_ang = (
            (
                (dynamics_params.w_alpha_r + dynamics_params.w_alpha_l)
                / dynamics_params.w1
            )
            / delta_time
        ) / 2
        kinematics_params = DynamicModelParameters(
            u1=1 / delta_time,
            u2=0,
            u3=0,
            w1=1 / delta_time,
            w2=0,
            w3=0,
            uar=kinematics_param_lin,
            ual=kinematics_param_lin,
            war=kinematics_param_ang,
            wal=kinematics_param_ang,
        )
        kinematics_params.wheel_distance = dynamics_params.wheel_distance
        kinematics_params.wheel_radius_left = dynamics_params.wheel_radius_left
        kinematics_params.wheel_radius_right = dynamics_params.wheel_radius_right
            
        return kinematics_params

    def _update_pos(self, action, delta_time: Optional[float] = None):
        """
        Update the position of the robot, simulating differential drive

        returns pos, angle
        """

        if delta_time is None:
            delta_time = self.delta_time

        if self.motion_model == MotionModelType.DYNAMICS:
            ...
        elif self.motion_model == MotionModelType.KINEMATICS:
            
            if self.is_delay_dynamics:
                dynamics_params = copy.deepcopy(self.state.state.parameters)
            else:
                dynamics_params = copy.deepcopy(self.state.parameters)
            
            kinematics_params = self.get_kinematics_params_from_dynamics_params(dynamics_params, delta_time)

            if self.is_delay_dynamics:
                self.state.state.parameters = kinematics_params
            else:
                self.state.parameters = kinematics_params

        else:
            raise ValueError(f"Invalid motion model: {self.motion_model}")

        action = DynamicsInfo(motor_left=action[0], motor_right=action[1])
        self.state = self.state.integrate(delta_time, action)
        q = self.state.TSE2_from_state()[0]
        pos, angle = self.weird_from_cartesian(q)
        pos = np.asarray(pos)

        # Update the parameters back to the original ones
        if self.motion_model == MotionModelType.KINEMATICS:
            if self.is_delay_dynamics:
                self.state.state.parameters = dynamics_params
            else:
                self.state.parameters = dynamics_params

        return pos, angle


def get_dir_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])


def get_right_vec(cur_angle: float) -> np.ndarray:
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])


def get_duckiebot_mesh(color: str) -> ObjMesh:
    change_materials: Dict[str, MatInfo]

    def get_duckiebot_color_from_colorname_extended(color: str) -> List[float]:
        colors = {
            "green": [0, 0.5, 0, 1],
            "red": [0.5, 0, 0, 1],
            "grey": [0.3, 0.3, 0.3, 1],
            "blue": [0.0, 0.0, 1.0, 1],
        }
        color = colors[color]
        return color

    color = np.array(get_duckiebot_color_from_colorname_extended(color))[:3]
    change_materials = {
        "gkmodel0_chassis_geom0_mat_001-material": {"Kd": color},
        "gkmodel0_chassis_geom0_mat_001-material.001": {"Kd": color},
    }
    return get_mesh("duckiebot", change_materials=change_materials)


def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of self.cur_pos is the center of rotation.
    """

    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec


def get_agent_corners(pos, angle):
    agent_corners = agent_boundbox(
        _actual_center(pos, angle),
        ROBOT_WIDTH,
        ROBOT_LENGTH,
        get_dir_vec(angle),
        get_right_vec(angle),
    )
    return agent_corners


class FrameBufferMemory:
    multi_fbo: int
    final_fbo: int
    img_array: np.ndarray
    width: int

    height: int

    def __init__(self, *, width: int, height: int):
        """H, W"""
        self.width = width
        self.height = height

        # that's right, it's inverted
        self.multi_fbo, self.final_fbo = create_frame_buffers(width, height, 4)
        self.img_array = np.zeros(shape=(height, width, 3), dtype=np.uint8)


def draw_axes():
    gl.glPushMatrix()
    gl.glLineWidth(4.0)
    gl.glTranslatef(0.0, 0.0, 0.0)

    gl.glBegin(gl.GL_LINES)
    L = 0.3
    gl.glColor3f(1.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(L, 0.0, 0.0)

    gl.glColor3f(0.0, 1.0, 0.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, L, 0.0)

    gl.glColor3f(0.0, 0.0, 1.0)
    gl.glVertex3f(0.0, 0.0, 0.0)
    gl.glVertex3f(0.0, 0.0, L)
    gl.glEnd()

    gl.glPopMatrix()
