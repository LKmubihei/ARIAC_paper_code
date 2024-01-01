#!/usr/bin/env python3


import os
import yaml
import random
import rclpy
from rclpy.node import Node
import time
import py_trees
import py_trees_ros
from py_trees.common import OneShotPolicy
from ament_index_python.packages import get_package_share_directory
from behaviors.floor_behaviors import grasp,place,move,flip
from ariac_msgs.srv import ExecutePlan
import threading
from geometry_msgs.msg import Pose,Point,Quaternion
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor


default_task_file = os.path.join(get_package_share_directory("paper_ws"), "config", "task.yaml")

plans=['move','grasp','move','place']

lock=threading.Lock()

## Test vision
class TreeBehavior(Node):
    def __init__(self):
        super().__init__("test_node")
        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()
        self.plans_service = self.create_service(ExecutePlan, 'execute_plan', self.handle_custom_service,callback_group=self.group1)
        self.timer = self.create_timer(0.3, self.timer_callback,callback_group=self.group2)  # 定时器周期为500ms
        self.receive_flag=False
        self.complete=False
        self.tree=None
        self.tree_state=None

    def timer_callback(self):
        if self.tree:
            # self.tree.tick_tock(period_ms=500.0)
            print("开始tick")
            self.tree.tick()
            root_status = self.tree.root.status
            
                # print("打印节点的状态:",root_status)


    def handle_custom_service(self, request, response):
        self.receive_flag=True
        self.tree=self.create_tree()
        print("树构建成功")
        while not self.complete:
            pass
        return response  
    # def generate_seq(self):
    #     seq = py_trees.composites.Sequence(name="manipulate", memory=True)
    #     seq.add_children(
    #         [
    #             move("move",'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),     
    #             grasp("grasp", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),          # 0抓  1放 2移动 3翻转 
    #            # place("place", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),         
    #         ])
    #     return seq

    def create_tree(self):
        """Create behavior tree with explicit nodes for each location."""

        root = py_trees.composites.Sequence(name="root", memory=True)
        seq = py_trees.composites.Sequence(name="manipulate", memory=True)
        # root = py_trees.decorators.OneShot(name="root",child=seq,policy=OneShotPolicy.ON_SUCCESSFUL_COMPLETION,)# 表示节点只有在成功完成时才会执行一次。
        tree = py_trees_ros.trees.BehaviourTree(root, unicode_tree_debug=False)
        tree.setup(timeout=15.0, node=self)

        # self.actions=[]
        # for plan in plans:
        #     action=self.create_action(plan.action_type,plan.part_type,plan.pose,self)
        #     seq.add_child(action)

        # seq.add_child(
        #     move("move",'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-0.0, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),     
        # )
        seq.add_children(
            [
                move("move",'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),     
                grasp("grasp", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),          # 0抓  1放 2移动 3翻转 
               # place("place", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),         
            ])
        root.add_child(seq)
        return tree
    
if __name__ == "__main__":
    rclpy.init()
    behavior = TreeBehavior()
    # rclpy.spin(behavior)
    tree_executor = MultiThreadedExecutor(num_threads=2)
    tree_executor.add_node(behavior)
    tree_executor.spin()

# class AutonomyBehavior_1(Node):
#     def __init__(self):
#         super().__init__("operate_node_1")

#         self.tree=self.create_tree()
#         self.tree_state=None


#     def create_tree(self):
#         """Create behavior tree with explicit nodes for each location."""

#         root = py_trees.composites.Sequence(name="root", memory=True)
#         seq = py_trees.composites.Sequence(name="manipulate", memory=True)
#         # root = py_trees.decorators.OneShot(name="root",child=seq,policy=OneShotPolicy.ON_SUCCESSFUL_COMPLETION,)# 表示节点只有在成功完成时才会执行一次。
#         tree = py_trees_ros.trees.BehaviourTree(root, unicode_tree_debug=False)
#         tree.setup(timeout=15.0, node=self)

        
#         # self.actions=[]
#         # for plan in plans:
#         #     action=self.create_action(plan.action_type,plan.part_type,plan.pose,self)
#         #     seq.add_child(action)

#         seq.add_children(
#             [
#                 move("move",'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),     
#                 grasp("grasp", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),          # 0抓  1放 2移动 3翻转 
#                 place("place", 'battery_blue',Pose(position=Point(x=-2.0799999285275055, y=-2.8050066689324376, z=0.7227856444671704), orientation=Quaternion(x=-0.0008636399229610038, y=-0.0028685476985478593, z=0.999995512348017, w=-3.965380435669542e-07)),tree.node),         
#             ])
#         root.add_child(seq)
#         return tree



    
#     def create_action(self, action_type, type, pose,node):
#         if action_type == "grasp":
#             return grasp(action_type, type, pose, node)
#         elif action_type == "move":
#             return move(action_type, type, pose, node)
#         elif action_type == "place":
#             return place(action_type, type, pose, node)
#         else:
#             raise ValueError(f"Unsupported action type: {action_type}")

#     def execute(self, period=0.5):
    
#         self.tree.tick_tock(period_ms=period * 500.0)
#         rclpy.spin(self.tree.node)
#         rclpy.shutdown()


# if __name__ == "__main__":
#     rclpy.init()
#     behavior = AutonomyBehavior_1()
#     behavior.execute()
#     # rclpy.spin(behavior)
