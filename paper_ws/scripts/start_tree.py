#!/usr/bin/env python3

import rclpy
import time
import threading
from rclpy.executors import MultiThreadedExecutor,SingleThreadedExecutor
from paper_ws.paper import PaperInterface
from paper_ws.behaviors import AGV_Move,StartCompetition

import py_trees
import py_trees_ros
from py_trees.common import Status,OneShotPolicy
from rclpy.node import Node



class Test(Node):
    def __init__(self):
        super().__init__("autonomy_node")
        self.tree=self.create_behavior_tree()

    def create_behavior_tree(self):

        root = py_trees.composites.Sequence(name="Topics2BB", memory=True)
        tree = py_trees_ros.trees.BehaviourTree(root, unicode_tree_debug=True)
        tree.setup(timeout=15.0, node=self)

        root.add_children([
            StartCompetition(tree.node),
            AGV_Move(1,1,tree.node),
        ])

        return tree

    def execute(self, period=0.5):
        self.tree.tick_tock(period_ms=period * 1000.0)
        rclpy.spin(self.tree.node)
        
        rclpy.shutdown()

if __name__ == "__main__":
    rclpy.init()
    behavior = Test()
    behavior.execute()
