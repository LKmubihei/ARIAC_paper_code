#!/usr/bin/env python3

import rclpy
from collections import deque
import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError 
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Duration
from rclpy.parameter import Parameter
from geometry_msgs.msg import Pose,PoseStamped,Vector3,Quaternion
from sensor_msgs.msg import JointState,Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from paper_ws.Kinematics import *
from paper_ws.interpolation import *
from math import *

from ariac_msgs.srv import *
from ariac_msgs.msg import *
from paper_ws.action import Grasp
from gazebo_msgs.srv import DeleteModel,SpawnEntity,DeleteEntity,SetModelState,GetModelState
from ariac_gazebo.spawn_params import (
    SpawnParams,
    RobotSpawnParams,
    SensorSpawnParams,
    PartSpawnParams,
    TraySpawnParams)

import py_trees
from action_msgs.msg import GoalStatus

gripper_states_ = {True: 'enabled',False: 'disabled'}


class GRASP_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,part_type,part_pose, node):
        super(GRASP_C, self).__init__(name)
        self.node = node
        self.part_type=part_type
        self.part_pose=part_pose                              # 0抓  1放 2移动 3翻转 

    def initialise(self):
        print("开始发送抓取消息")
        self.grasp_part_action=ActionClient(self.node, Grasp, 'grasp_2')
        self.grasp_part_action.wait_for_server()
        self.goal_status=None   
        goal_msg = Grasp.Goal()
        
        goal_msg.type = self.part_type
        goal_msg.pose = self.part_pose
        goal_msg.dest='None'
        print("执行抓取的命令位置是:",self.part_pose.position,"pose的id是",id(self.part_pose))
        print("打印一下我要发送的位置",goal_msg.pose.position)  # y=-2.625  ，实际位置在-2.8050
        goal_msg.number= 0
        self.grasp_send_goal_future = self.grasp_part_action.send_goal_async(goal_msg)
        self.grasp_send_goal_future.add_done_callback(self.goal_response_callback)
        

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.info('Goal rejected :(')
            return

        self.logger.info('Goal accepted :)')

        self.grasp_get_result_future = goal_handle.get_result_async()
        self.grasp_get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.goal_status = future.result().status
        print("成功执行抓取")

    def update(self):

        if self.goal_status is not None:
            if self.goal_status == GoalStatus.STATUS_SUCCEEDED:
                print("在GraspPart这成功")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.grasp_part_action = None

class PLACE_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,part_type,part_pose, dest,node ):
        super(PLACE_C, self).__init__(name)
        self.node = node
        self.part_type=part_type
        self.part_pose=part_pose                            # 0抓  1放 2移动 3翻转 
        self.dest=dest
        self.goal_status=None

    def initialise(self):
        self.place_part_action=ActionClient(self.node, Grasp, 'grasp_2')
        self.place_part_action.wait_for_server()
        self.goal_status=None   
        goal_msg = Grasp.Goal()
                     
        goal_msg.type = self.part_type
        goal_msg.pose = self.part_pose
        goal_msg.number= 1
        goal_msg.dest=self.dest
        self.grasp_send_goal_future = self.place_part_action.send_goal_async(goal_msg)
        self.grasp_send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.info('Goal rejected :(')
            return

        self.logger.info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.goal_status = future.result().status

    def update(self):

        if self.goal_status is not None:
            if self.goal_status == GoalStatus.STATUS_SUCCEEDED:
                print("在PlacePart 这成功")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.place_part_action = None

class MOVE_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,part_type,part_pose,dest, node ):
        super(MOVE_C, self).__init__(name)
        self.node = node
        self.part_type_1=part_type
        self.part_pose_1=part_pose                             # 0抓  1放 2移动 3翻转 
        self.dest_1=dest
        self.count=0

    def initialise(self):
        self.move_action=ActionClient(self.node, Grasp, 'grasp_2')
        self.move_action.wait_for_server()
        self.goal_status=None   
        goal_msg = Grasp.Goal()
        print("移动消息已发送",self.dest_1)
                     
        goal_msg.type = self.part_type_1
        # self.part_pose_1.position.y= floor_positions[self.dest_1][1]
        goal_msg.pose = self.part_pose_1
        goal_msg.number= 2
        goal_msg.dest= self.dest_1
        self.grasp_send_goal_future = self.move_action.send_goal_async(goal_msg)
        self.grasp_send_goal_future.add_done_callback(self.goal_response_callback)
        print("打印一下我要发送的移动位置",goal_msg.pose.position)  # y=-2.625  ，实际位置在-2.8050

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.info('Goal rejected :(')
            return

        self.logger.info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.goal_status = future.result().status

    def update(self):
        self.count= self.count+1
        if self.goal_status is not None:
            if self.goal_status == GoalStatus.STATUS_SUCCEEDED:
                print(" move成功,返回SUCCESS")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        # print("现在 self.goal_status:", self.goal_status,"第", self.count)
        # print(" 正在移动中")
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.move_action = None

class FLIP_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,part_type,part_pose, node ):
        super(FLIP_C, self).__init__(name)
        self.node = node
        self.part_type=part_type
        self.part_pose=part_pose                              # 0抓  1放 2移动 3翻转 
        self.goal_status=None

    def initialise(self):
        self.flip_part_action=ActionClient(self.node, Grasp, 'grasp_2')
        self.flip_part_action.wait_for_server()
        self.goal_status=None   
        goal_msg = Grasp.Goal()
                     
        goal_msg.type = self.part_type
        goal_msg.pose = self.part_pose
        goal_msg.number= 3
        goal_msg.dest='None'
        self.grasp_send_goal_future = self.flip_part_action.send_goal_async(goal_msg)
        self.grasp_send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.info('Goal rejected :(')
            return

        self.logger.info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.goal_status = future.result().status

    def update(self):

        if self.goal_status is not None:
            if self.goal_status == GoalStatus.STATUS_SUCCEEDED:
                print("在PlacePart 这成功")
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.flip_part_action = None

class Robot_Normal_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,node):
        super(Robot_Normal_C, self).__init__(name)
        self.node = node
        self.floor_is_enabled=True

    def initialise(self):
        self.robot_health_subscriber = self.node.create_subscription(Robots, '/ariac/robot_health', self.robot_health_state_callback, qos_profile_sensor_data)

    def update(self):
        
        if self.floor_is_enabled:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.robot_health_subscriber =None

    def robot_health_state_callback(self, msg):
        self.floor_is_enabled=msg.floor_robot
        # print("打印一下接受的消息",self.floor_is_enabled)

class Gripper_State_C(py_trees.behaviour.Behaviour):

    def __init__(self, name,node):
        super(Gripper_State_C, self).__init__(name)
        self.node = node
        self.ceiling_robot_gripper_state=None

    def initialise(self):
        self.floor_robot_gripper_state_sub = self.node.create_subscription(VacuumGripperState, 
                    '/ariac/ceiling_robot_gripper_state', self.ceiling_robot_gripper_state_cb, qos_profile_sensor_data)
    def update(self):
        
        if self.ceiling_robot_gripper_state:
            if self.ceiling_robot_gripper_state.attached:
                return py_trees.common.Status.SUCCESS
            else:
                return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.SUCCESS


    def terminate(self, new_status):
        self.logger.info(f"Terminated with status {new_status}")
        self.ceiling_robot_gripper_state =None

    def ceiling_robot_gripper_state_cb(self, msg: VacuumGripperState):
        self.ceiling_robot_gripper_state = msg
        print("ceiling夹爪的状态",self.ceiling_robot_gripper_state)


