# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import click
import cv2
import numpy as np

from stretch.agent import RobotClient

# Mapping and perception
from stretch.core.parameters import get_parameters
from stretch.dynav import RobotAgentMDP
from code_as_policy import RobotCodeGen





@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--server_ip", default="", type=str, help="IP address for the MDP agent")
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=-1)
@click.option("--re", default=3, type=int, help="Choose between stretch RE1, RE2, RE3")
@click.option("--method", default="dynamem", type=str)
@click.option("--env", default=1, type=int)
@click.option("--test", default=1, type=int)
@click.option(
    "--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)"
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value 'output.npy'",
)
def main(
    server_ip,
    manual_wait,
    navigate_home: bool = False,
    explore_iter: int = 5,
    re: int = 1,
    method: str = "dynamem",
    env: int = 1,
    test: int = 1,
    input_path: str = None,
    robot_ip: str = "10.19.199.94",
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """
    click.echo("Will connect to a Stretch robot and collect a short trajectory.")
    robot = RobotClient(robot_ip=robot_ip)
    robot.move_to_nav_posture()

    print("- Load parameters")
    parameters = get_parameters("dynav_config.yaml")
    # print(parameters)
    if explore_iter >= 0:
        parameters["exploration_steps"] = explore_iter
    object_to_find, location_to_place = None, None
    robot.move_to_nav_posture()
    robot.set_velocity(v=30.0, w=15.0)

    print("- Start robot agent with data collection")
    demo = RobotAgentMDP(
        robot, parameters, server_ip=server_ip, re=re, env_num=env, test_num=test, method=method
    )

    if input_path is None:
        demo.rotate_in_place()
    else:
        demo.image_processor.read_from_pickle(input_path)

    demo.save()
    code_as_poicy = RobotCodeGen(robot, demo)

    while True:
        code_as_poicy.task_complete(input("Task: "))
        demo.save()

if __name__ == "__main__":
    main()
