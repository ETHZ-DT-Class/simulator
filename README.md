# Duckietown simulation environment and ROS bridge

## Do you need to read this README?
For the development of the exercises, you **don't** need to read this README. The exercises will provide you with all the necessary information to run the simulator with the correct parameters. This README is for those who want to know more about the simulator and its parameters (e.g., to [reduce computational load](#reduce-computational-load-of-the-simulation)).

## Easy start
### Clone the repository
If the simulator repository is not already present at [user_code_mount_dir/simulator/](../), clone the repository.

```
git clone https://github.com/ETHZ-DT-Class/simulator.git /home/duckie/vnc-docker/user_code_mount_dir/simulator
```
### Run

***IMPORTANT***: in theory, you should NEVER run the simulator using the simple `roslaunch simulator simulator.launch` command while developing the exercises, because it will launch the simulator loading the default config parameters. When an exercise requires to run the simulator, the exercise will provide a specific way to launch the simulator such that the correct exercise-specific or task-specific config parameters are loaded. Check the README of the exercise for more information.

The simulator is a standard ROS package: inside the container, build the package running `catkin build` inside the folder `/code/catkin_ws/` and source the ROS workspace with `source /code/catkin_ws/devel/setup.bash`. Then, run the simulator node with `roslaunch simulator simulator.launch`, which will run it loading the default config parameters. If you want to change the default config parameters, modify the files in the [assets/simulator/simulator/params](./params) folder. More information about changing the default parameters can be found in the [Parameters](#parameters) section.

You can visualize the simulation with RQT and/or Rviz.

### Updates

During the class, the simulator could be updated with new feature or bug fixes. If this is the case, we will ask you to update the git repository of the simulator and to work with the new version of the simulator only. To do so, you have to `git pull` while being inside the simulator folder. Then you may run `catkin build` inside the docker container to build the updated simulator package.

You ***DO NOT*** need to stop and remove the current running container when updating the simulator repository, or to pull/build a new version of the docker image. You can continue to work with the same running container. You just have to update the git repository of the simulator.

We will update on Moodle the version number of the simulator you should use, to help you confirm that you have correctly updated it. To check this, simply run `make show-versioning` *outside* the container, or manually check the *version* tag inside the `package.xml` file of the simulator repository.

**IMPORTANT** ROS packages have a version with the format *X.Y.Z* ([semantic versioning](https://semver.org/)). For this class, the version will only change in its X and Y values, with the following meaning:
- increase in X value (major update): we updated the code in some way, or anyway there is an important update, you **must** update the package. 
- increase in Y value (minor upddate): we updated the instructions/documentation only (comments in code, or READMEs, ...), you don't have to update the repository (but is highly recommended)



## Details
The simulator is a ROS package that provides a custom Duckietown simulation environment and a custom ROS bridge. The simulation environment is based on the [duckietown-gym](https://github.com/duckietown/gym-duckietown) repository and is used to simulate Duckietown environments and Duckiebot sensors. The ROS bridge is used to communicate between the simulation environment and ROS (convert ROS messages to simulation commands, and simulation observations to ROS messages)

## Simulation environment

The simulation can be displayed in RQT or RViz visualizing the published 'external window' topic. There are different modes: `human`, `top_down`, `free_camera`. It can also not be displayed at all to reduce computational load (see [#Reduce computational load of the simulation](#reduce-computational-load-of-the-simulation)).

The simulation frame rate update step is independent on the real time frequency update of the simulation. I.e., the simulated dynamics and sensors can run at 100 Hz, so a simulated time step of 0.01 s, but the actual internal "update environment" function can be called at whatever real time frequency the user preferred (based on the specified real time factor). This allows to adjust the computational load based on the hardware.

### Parameters

Default parameters are provided in the [assets/simulator/simulator/params](./params) folder.

***IMPORTANT***: you don't have to modify these default parameters while developing the exercises. When an exercise requires to run the simulator, the exercise will provide a specific way to launch the simulator such that the correct exercise-specific or task-specific config parameters are loaded. Check the README of the exercise for more information.

You can define your own parameter folder and load it by setting in the launch file [assets/simulator/simulator/launch/simulator.launch](./launch/simulator.launch) the parameter `package_param_path` to the path of your parameter folder (`roslaunch simulator simulator.launch package_param_path:=<YOUR_FOLDER_PATH>`)

The simulation parameters are divided into different files:

<details><summary>bridge.yaml</summary>

[bridge.yaml](./params/bridge.yaml)

```
realtime_factor: [int, float] # speed up the simulation by the factor (2.0 sim is updated 2x faster than real time, but 2x computational load)
repeat_action_until_new: bool # held the action until a new action is received, otherwise stop the robot until new action

reset: # reset options
  mode: str # [reset | kill] - reset the simulation or kill the node when the reset is called
  reset_ros_time_when_env_is_reset: bool # the ros time is reset when the environment is being reset, i.e. topics' stamp will reset to 0
  reset_action_when_env_is_reset: bool # the action is reset when the environment is being reset, i.e. the robot will start still

print_startup_log: bool # print the startup log of the simulation

meshes_folder_path: str # path for RViz to search for the meshes

post_startup_log_text: str # print a special string of your choice, can be useful to understand which parameter folder is being used
```
</details>

<details><summary>logger.yaml</summary>

[logger.yaml](./params/logger.yaml)

```
level: str # [debug_timing | debug | info | warning | error | critical]
```
</details>

<details><summary>simulator_obs.yaml</summary>

[simulator_obs.yaml](./params/simulator_obs.yaml)

```
camera:
  enable: bool
  rate: [int, float]
  width: int
  height: int
  compression_format: str # [jpeg | png]
  
imu:
  enable: bool
  rate: [int, float]
  is_rotated: bool # if true, the imu data will be rotated 90 degrees counter-clockwise, as the imu on the real Duckiebot

wheel_encoders:
  enable: bool
  rate: [int, float]
  ticks_per_revolution: int

gt_pose:
  enable: bool
  rate: [int, float]

lane_pose:
  enable: bool
  rate: [int, float]

command:
  enable: bool
  rate: [int, float]


noise: # add noise n_t to the data, obtained from white noise wn and bias b: n_t = b_mu,t + N(0, wn_sigma), with b_mu,t = b_mu,t-1 + N(0, b_sigma)

  imu:
    enable: bool

    lin_acc: # noise of the x and y axis linear acceleration
      white_noise:
        sigma: [int, float]
      bias:
        mu: [int, float]
        sigma: [int, float]

    ang_vel: # noise of the z axis angular velocity
      white_noise:
        sigma: [int, float]
      bias:
        mu: [int, float]
        sigma: [int, float]

    orientation: # noise of the z axis orientation
      white_noise:
        sigma: [int, float]
      bias:
        mu: [int, float]
        sigma: [int, float]

  wheel_encoders: # the noise is directly applied to wheel rotation radians, NOT to encoder ticks
    enable: bool

    left:
      white_noise:
        sigma: [int, float]
      bias:
        mu: [int, float]
        sigma: [int, float]

    right:
      white_noise:
        sigma: [int, float]
      bias:
        mu: [int, float]
        sigma: [int, float]
```
</details>

<details><summary>simulator_env.yaml</summary>

[simulator_env.yaml](./params/simulator_env.yaml)

```
seed: int # seed for random number generation; 0 means random seed
frame_rate: [int, float] # inner frame rate of the simulation, i.e. sim delta time step is 1/frame_rate
map_name: str # duckietown map name
max_steps: int # maximum number of steps before reset (mostly for RL)
domain_rand: bool # domain randomization
accept_start_angle_deg: [int, float] # max angle difference between start agent direction and lane direction in degrees 
full_transparency: bool # RL only, return the full state of the simulator
distortion: bool # enable camera distortion
additional_keyboard_control: bool # additional keyboard control together with published command topic (keyboard control has precedence) - currently not supported
duckiebot_color: str # [blue | red] - duckiebot color for top_down rendering and Rviz mesh

init_pose: # initial agent pose
  enable: bool # use the specified initial pose, otherwise use the pose provided by the map config
  x: [int, float] # agent initial x position in meters
  y: [int, float] # agent initial y position in meters
  theta: [int, float] # agent initial orientation in radians

motion_model:
  type: str # [dynamics | kinematics] - dynamics is more realistic, kinematics has immediate reaction
  params:
    delay: [int, float] # delay in seconds in the agent's actions
    wheel_distance: [int, float] # distance between the two wheels in meters - original should be 0.102
    wheel_radius_left: [int, float] # radius of the left wheel in meters - original should be 0.0335
    wheel_radius_right: [int, float] # radius of the right wheel in meters - original should be 0.0335
  applied_command_noise: # add noise before applying the command to the agent
    enable: bool
    min_action_threshold: [int, float] # no noise if abs(action) < threshold
    white_noise:
      sigma: [int, float]
    bias:
      mu: [int, float]
      sigma: [int, float]

update: # choose your preferred update method and specify its value
  use: str # [frame_rate | delta_time_step] - update the physics using the specified frame rate or delta time step
  frame_rate: [int, float] # inner frame rate of the physics update in Hz, i.e. sim delta time step is 1/frame_rate seconds
  delta_time_step: [int, float] # inner delta time step of the physics update in seconds, i.e. frame rate is 1/delta_time_step Hz

rendering:
  frustum_filtering: bool # activate frustum culling
  frustum_filtering_min_arccos_threshold: [float, int] # min arccos threhsold for the frustum filtering
  depth_filtering_factor: [int, float] # filter objects far away from the agent (depth > depth_filtering_factor * tile_size)
  depth_texture_resize_factor: [int, float] # resize the texture for objects far away from the agent (depth > depth_filtering_factor * tile_size)
  distant_texture_resize_factor: [int, float] # texture resize factor (multiple of 2 or divisor of 1/2)
  force_texture_resize_floor: bool # force the floor texture to be resized always
  texture_style: str # [photos | photos-segmentation | segmentation | smooth | synthetic]
  skip_frame_buffer_copy: bool # skip the frame buffer copy (faster on CPU, should not create problems apart from small 3D objects artifacts due to no depth buffer)

compute_reward: bool # compute RL reward

check: # reset the simulation if the checks are not satisfied
  drivable: bool # check if the agent is on a drivable surface, otherwise reset
  collisions: bool # check if the agent has collided, otherwise reset

display: # external displaying of the simulation
  screen_enable: bool # display the external simulation rendering on screen - currently not supported
  topic_enable: bool # publish the external simulation rendering on a topic
  mode: str # [human | top_down | free_camera]
  width: int # image width
  height: int # image height
  compression_format: str # [jpeg | png]
  rate: int # update rate
  reuse_camera_obs_if_possible: bool # reuse the camera observation if available and same scale, and mode is not top_down
  enable_segmentation: bool # render as segmented image
  show: # display the following information on top of the image
    pose: bool # display the agent's pose
    speed: bool # display the agent's speed
    steps: bool # display n. simulation steps
    time_stamp: bool # display simulation time stamp
```
</details>

<details><summary>topics.yaml</summary>

[topics.yaml](./params/topics.yaml)

```
sub: # Subscribed topics
  wheel_cmd: str

pub: # Published topics
  camera: str
  imu: str
  wheel_encoder_left: str
  wheel_encoder_right: str
  gt_pose: str
  lane_pose: str
  simulation_external_window: str
  command: str

srv: # Services (server)
  reset_sim: str
  kill_sim: str
```
</details>

##### Special Python values
If you want to use special Python values like `float("inf")`, `float("-inf")`, or `None`, just write the corresponding string `inf`, `-inf`, and `None`. However, if you want to use the string `"inf"`, `"-inf"`, or `"None"` as it is, this is not supported.

### Input command

The simulated duckiebot is controlled by commanding the left and right wheels with torques inputs normalized in the range [0, 1]. The ROS bridge is subscribed to [duckietown_msgs/WheelsCmdStamped](https://github.com/duckietown/dt-ros-commons/blob/daffy/packages/duckietown_msgs/msg/WheelsCmdStamped.msg) message via the topic described in [params/topics.yaml](./params/topics.yaml). Publish the desired wheel commands to this topic to control the simulated duckiebot.

### Output sensors

The simulated sensors options are managed in the param files [params/simulator_obs.yaml](./params/simulator_obs.yaml) (status, rate, noise) and [params/topics.yaml](./params/topics.yaml) (published topic names).

#### Sensors
You can choose to enable the following sensors (and their update rates) in the simulation environment:
- Camera
- IMU
- Wheel encoders

Moreover, you can enable the following "additional sensors" (no noise is added to these "measurements"):
- Ground truth pose (ground truth global pose of the duckiebot)
- Ground truth lane pose (ground truth local pose of the duckiebot in the lane)
- Input command (just mirroring the received input command)

#### Noise model
For the IMU and wheel encoders, you can also add noise $n_t$ to the sensor data, obtained from a white noise $wn$ and a varying bias $b$: $n_t = \mu_{b,t} + \mathcal{N}( 0 , \sigma_{wn} )$ with $\mu_{b,t} = \mu_{b,t-1} + \mathcal{N}( 0 , \sigma_{b} )$. Noise can also be added to the input command before applying it to the agent.

### Services

The ROS bridge provides the following (server) services to interact with the simulation environment (for the actual services' names, check/modify [params/topics.yaml](./params/topics.yaml)):
- `reset_sim`: reset the simulation environment (reset independent of the `reset/mode` parameter in [params/bridge.yaml](./params/bridge.yaml))
- `kill_sim`: kill the simulation node


## Reduce computational load of the simulation
Some tips to reduce the computational load of the simulation:
- Rendering (in [params/simulator_env.yaml](./params/simulator_env.yaml)):
  - Enable frustum filtering (`rendering: frustum_filtering`) and increase the minimum arccos threshold (`rendering: frustum_filtering_min_arccos_threshold`)
  - Decrease the depth filtering factor (`rendering: depth_filtering_factor`)
  - Decrease the depth texture resize factor (`rendering: depth_texture_resize_factor`)
  - Decrease the distant texture resize factor (`rendering: distant_texture_resize_factor`)
  - Enable the forcing of the floor texture resize (`rendering: force_texture_resize_floor`)
  - Enable the skipping of the frame buffer copy (`rendering: skip_frame_buffer_copy`) - more useful when running on CPU than GPU
  - Disable the external displaying of the simulation and/or camera observation (see next points)
- External displaying of the simulation (in [params/simulator_env.yaml](./params/simulator_env.yaml)):
  - Reduce the display rate (`display: rate`) - don't worry, the display rate is NOT related to the actual camera observation rate of the camera topic
  - Reduce the display resolution (`display: width`, `display: height`) - don't worry, the display resolution is NOT related to the actual camera observation resolution of the camera topic
  - Enable the reuse of the already computed camera observation (`display: reuse_camera_obs_if_possible`)
  - Disable the display completely (`display: topic_enable`)
- Camera observation (in [params/simulator_obs.yaml](./params/simulator_obs.yaml)):
  - Do not subscribe to the camera topic and do not visualize the camera topic in RQT (if nothing is subscribed to the camera topic, the camera observation is not rendered)
  - Disable the camera observation completely (`camera: enable`)
- Real time factor (in [param/sbridge.yaml](./params/bridge.yaml)):
If the simulation is still using too much computational resources, you can reduce the real time factor `realtime_factor`. The simulated dynamics and sensors will still run at the same specififed simulated framerate or delta time step (`update: frame_rate` or `update: delta_time_step` in [param/simulator_env.yaml](./params/simulator_env.yaml)). This will reduce the computational load of the simulation, but from the external point of view, the simulation will run slower. But don't worry, from the point of view of the simulated robot, nothing changed.