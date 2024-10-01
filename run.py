#!/usr/bin/env python3

import rospy
import time

from src.SimulatorRosBridge import SimulatorRosBridge


def main():
    rospy.init_node("Exercise_Simulator", anonymous=True)

    simulator_ros_bridge = SimulatorRosBridge()

    # Testing everything runs
    simulator_ros_bridge.enable_test_mode()
    n_test_steps = 2
    idx_test_step = 0
    while not rospy.is_shutdown():
        simulator_ros_bridge.test_step()
        idx_test_step += 1
        if idx_test_step > n_test_steps:
            break
        simulator_ros_bridge.sleep()
    simulator_ros_bridge.disable_test_mode()

    # Actual simulation
    while not rospy.is_shutdown():
        simulator_ros_bridge.step()
        simulator_ros_bridge.sleep()


if __name__ == "__main__":
    main()
