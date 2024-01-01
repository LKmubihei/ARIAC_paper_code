#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from std_msgs.msg import Float64,String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointControllerState,JointTrajectoryControllerState
from paper_ws.Kinematics import *
from paper_ws.interpolation import *
from math import *
import copy
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from ariac_msgs.srv import *
from ariac_msgs.msg import *
from paper_ws.data import *
from paper_ws.action import Grasp

class FloorActionServer(Node):
    gripper_states_ = {True: 'enabled',False: 'disabled'}

    def __init__(self):
        super().__init__('bandit_action_server')

        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()
        self.group3 = MutuallyExclusiveCallbackGroup()
        self.group4 = MutuallyExclusiveCallbackGroup()   
        self.group5 = MutuallyExclusiveCallbackGroup()    
        self.group6 = MutuallyExclusiveCallbackGroup()  
        self.group7 = MutuallyExclusiveCallbackGroup()    
        self.kitting_arm_one_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/floor_robot_controller/state',
                                self.kitting_arm_one_joint_state_callback, qos_profile_sensor_data,callback_group=self.group1)
        self.floor_one_action_client=ActionClient(self, FollowJointTrajectory, '/floor_robot_controller/follow_joint_trajectory',callback_group=self.group2)
        self.floor_robot_gripper_one_state = VacuumGripperState()
        self.floor_gripper_one_enable = self.create_client(VacuumGripperControl, "/ariac/floor_robot_enable_gripper",callback_group=self.group3)
        self.floor_robot_gripper_one_state_sub = self.create_subscription(VacuumGripperState, '/ariac/floor_robot_gripper_state', 
                                            self.floor_robot_gripper_one_state_cb,  qos_profile_sensor_data,callback_group=self.group4)

        self.linear_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/linear_rail_controller/state', self.linear_joint_state_callback, 
                                                                      qos_profile_sensor_data,callback_group=self.group6)
        self.linear_action_client=ActionClient(self, FollowJointTrajectory, '/linear_rail_controller/follow_joint_trajectory',callback_group=self.group7)

        self._action_server = ActionServer(self,Grasp,'grasp_1',self.execute_callback,callback_group=self.group5)
        self.get_logger().info('Action server created...')


        self.kitting_arm_one_joint_states=None
        self.kitting_arm_one_joint_names = [
            'floor_shoulder_pan_joint','floor_shoulder_lift_joint',  'floor_elbow_joint',
            'floor_wrist_1_joint', 'floor_wrist_2_joint','floor_wrist_3_joint'
        ]
        self.linear_joint_names = ['linear_actuator_joint']
        self.kitting_typical_joints = {
            "init_state" : [1.4820999965423397-pi/2, -1.6356888311746864, 1.9210404979505746, -1.8276216909939889, -1.5708049327960403, -3.1302637783910976],
            
            "standby" : [1.5820963319369694, -1.6356888311746314, 1.921040497950596, -1.8276216909939276, -1.5708049327979872, -3.1302637783918614],
            "standby2" : [1.5820963319369694-pi, -1.6356888311746314, 1.921040497950596, -1.8276216909939276, -1.5708049327979872, -3.1302637783918614],
            "bin_agv_insert_joint": [-9.908213582932035e-06, -1.6356881698442969, 1.9210396904708134, 4.434232672827393, -1.570804295066237, -3.130262516117118],
            "flip_init_state" : [3.139979598681924, -1.0823125299292018, 1.7835319716553002, 5.542528446925819, -1.4273425719694686 - pi/2, -3.1399976775082745],
            "conveyor_insert": [0.8530278322946581+pi*0.8, -0.6160195671945434-0.8, 1.4189652905293846+0.9, 0.7678673579752529, 1.5707984502869392, -0.7177684944226037],
            "machine":[0.8530278322946581+pi*1, -0.6160195671945434-0.8, 1.4189652905293846+0.9, 0.7678673579752529, 1.5707984502869392, -0.7177684944226037],
        }  

        self.spin_lock = threading.Lock()
        self.init_rotation = numpy.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00]])
        self.kitting_base_x = -1.30 
        self.kitting_base_y = 0.0
        self.kitting_base_z = 0.93+0.05     # 加上基座的高度
        self.count=0


    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Process the goal
        goal = goal_handle.request
        self.get_logger().info(f'Received goal: type={goal.type}, pose={goal.pose}')

        if goal.number==0:                                   # 0抓  1放 2移动 3翻转 
            self.kitting_robot_one_init('init_state')
            print("打印一下我收到的位置0",goal.pose.position)
            self.grasp_part(goal.type, goal.pose)
        elif goal.number==1:                                   # 
            self.place_part(goal.type, goal.pose)
        elif goal.number==2:     
            goal_msg = deepcopy(goal.pose)
            goal_msg.position.y= floor_positions[goal.dest][1]                                # 
            self.move_to( goal_msg)

        elif goal.number==3:                                     # 
            self.flip_part(goal.type, goal.pose)


        # Perform some processing and set the result
        goal_handle.succeed()
        result = Grasp.Result()
        result.success = True
        print("对于",goal.number,"我已经发送:",result)

        return result

    def grasp_part(self,type, pose):
        target_matrix =self.Pose2Robot(pose) 
        print("打印一下我收到的位置",pose.position)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.init_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[2,3]=p2[2,3]+0.2

        self.set_floor_robot_gripper_state(True)
        self.MOV_M_one(p2,eps =0.01)  
        
        p1[2,3] = target_matrix[2,3]+part_heights[type]+0.01 
        self.MOV_M_one(p1,eps =0.01,times=5)  
        sleep(0.3)

        repick_nums = 0
        while not self.floor_robot_gripper_one_state.attached:
            if repick_nums >= 10:
                break
            
            repick_nums = repick_nums + 1
            p1[2,3]=p1[2,3]-0.002
            self.MOV_M_one(p1,eps =0.01,times=5)  
            sleep(0.2)

        p1[2,3]=p1[2,3]+0.4
        self.MOV_M_one(p1,eps =0.01)    


    def place_part(self,type, pose):
        target_matrix =self.Pose2Robot(pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.init_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[2,3]=p2[2,3]+0.3
        self.MOV_M_one(p2,eps =0.01)  
        sleep(0.2)

        p1[2,3] = target_matrix[2,3]+part_heights[type]+0.1
        self.MOV_M_one(p1,eps =0.01,times=5)  
        sleep(0.3)
        self.set_floor_robot_gripper_state(False)    
        p1[2,3]=p1[2,3]+0.4
        self.MOV_M_one(p1,eps =0.01)  

    def flip_part(self,type, pose):
        sleep(3)
        # target_matrix =self.Pose2Robot(pose) 
        # position,rpy = Matrix2Pos_rpy(target_matrix)
        # target_matrix = Rot2Matrix(self.init_rotation, position)

        # p1 = copy.deepcopy(target_matrix)
        # p2 = copy.deepcopy(target_matrix)

        # p2[2,3]=p2[2,3]+0.2

        # self.set_floor_robot_gripper_state(True)
        # self.MOV_M_one(p2,eps =0.01)  
        
        # p1[2,3] = target_matrix[2,3]+part_heights[type]+0.01 
        # self.MOV_M_one(p1,eps =0.01,times=5)  
        # sleep(0.3)

        # repick_nums = 0
        # while not self.floor_robot_gripper_one_state.attached:
        #     if repick_nums >= 10:
        #         break
            
        #     repick_nums = repick_nums + 1
        #     p1[2,3]=p1[2,3]-0.002
        #     self.MOV_M_one(p1,eps =0.01,times=5)  
        #     sleep(0.2)

        # p1[2,3]=p1[2,3]+0.4
        # self.MOV_M_one(p1,eps =0.01)  

    def move_to(self,pose):
        '''
        到达指定位置点 e.g., bin1
        '''
        #获取目标点的位置，主要用Y轴做移动
        end_point =-pose.position.y
        q_begin = self.kitting_base_y 
        distance = abs(q_begin - end_point)
        move_time = distance/kitting_velocity
        q_begin = [q_begin]
        q_end= [end_point]   
        traj = traj_generate(self.linear_joint_names,q_begin,q_end,move_time)
        self.move(self.linear_action_client,traj)
        print("看一下移动时间",move_time," 开始位置:",q_begin," 要求位置:",end_point)
        sleep(move_time)

    def kitting_arm_one_joint_state_callback(self, msg):
        self.kitting_arm_one_joint_states =msg.actual.positions
        self.count+=1
        # print(self.count)
        # if(self.count%5==0):
        #     print("msg.position",msg.actual.positions)

    def linear_joint_state_callback(self, msg):
        self.kitting_base_y = msg.actual.positions[0]

    def floor_robot_gripper_one_state_cb(self, msg: VacuumGripperState):
        self.floor_robot_gripper_one_state = msg

    def set_floor_robot_gripper_state(self, state) -> bool: 
        if self.floor_robot_gripper_one_state.enabled == state:
            self.get_logger().warning(f'Gripper is already {self.gripper_states_[state]}')
            return
        
        request = VacuumGripperControl.Request()
        request.enable = state
        
        future = self.floor_gripper_one_enable.call_async(request)
        
        # try:
        #     with self.spin_lock:
        #         rclpy.spin_until_future_complete(self, future)
        # except KeyboardInterrupt:
        #     raise KeyboardInterrupt
        sleep(0.3)
        
        if future.result() is not None:
            response = future.result()
            if response.success:
                 self.get_logger().info(f'Changed gripper state to {self.gripper_states_[state]}')
            else:
                self.get_logger().warning('Unable to change gripper state')
        else:
            self.get_logger().info('Service call failed.')


    def kitting_robot_one_init(self, state,Wait_flag = True): 
        while not self.kitting_arm_one_joint_states and rclpy.ok():
            sleep(0.1)
            print('waiting for initialization 卡 ...')

        # 复位
        init_position = copy.deepcopy(self.kitting_typical_joints[state])
        self.MOV_A_one(init_position, eps =0.02,sleep_flag = Wait_flag)
        print ('initialization success !')
        return True

    def MOV_A_one(self, target_joint_angles,time_from_start = 0.0,eps = 0.01, sleep_flag=True):         #输入的是关节角度向量Angle，直接控制运动
        '''
        time_from_start,默认按照最快时间执行
        eps = 0.0025
        '''
        q_begin = copy.deepcopy(self.kitting_arm_one_joint_states)
        q_end = target_joint_angles
        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        if time_from_start == 0.0:
            time_from_start = max(delta)/kitting_angle_velocity
        #做运动插补
        traj = traj_generate(self.kitting_arm_one_joint_names,q_begin,q_end,time_from_start)
        ##print traj

        self.move(self.floor_one_action_client,traj)
        sleep(time_from_start)

    def MOV_M_one(self, target_matrix,eps = 0.005,flip_flag=False,times=1.5,time_set=0.0):  #输入的是目标变换矩阵Mat,比MOV_A多求逆解，目标抓取

        q_begin = copy.deepcopy(self.kitting_arm_one_joint_states)
        # print("现在看一下q_begin:",q_begin)

        #求逆解
        # print("MOV_M_one--target_matrix",target_matrix)
        target = IKinematic(target_matrix,q_begin,A_k)
        if target==None:
            print("解不出来,你看看target_matrix",target_matrix)
            return False
        # #print "IKinematic_result:",target
        if flip_flag:
            target[-1] = q_begin[-1]
        #求时间
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        ##print delta
        distance = max(delta)
        if time_set==0.0:
            time_from_start = distance/kitting_angle_velocity*times
        else:
            time_from_start = time_set
        traj = traj_generate(self.kitting_arm_one_joint_names,q_begin,target,time_from_start)
        self.move(self.floor_one_action_client,traj)
        sleep(time_from_start)

    def move(self,action_client,goal_trajectory): 
            
        goal_msg = FollowJointTrajectory.Goal()

        # Fill in the trajectory message
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = goal_trajectory.joint_names
        goal_msg.trajectory.points = goal_trajectory.points
        
        # future.add_done_callback(self.goal_response_callback) 
        action_client.wait_for_server()
        self._floor_robot_send_goal_future = action_client.send_goal_async(
            goal_msg)
        
        self._floor_robot_send_goal_future.add_done_callback(
            self.floor_robot_goal_response_callback)

    def floor_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._floor_robot_get_result_future = goal_handle.get_result_async()
        self._floor_robot_get_result_future.add_done_callback(
            self.floor_robot_get_result_callback)

    def floor_robot_get_result_callback(self, future):
        result = future.result().result
        result: FollowJointTrajectory.Result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info("Move succeeded")
        else:
            self.get_logger().error(result.error_string)

        self.floor_robot_at_home = True

    def Pose2Robot(self,pose):
        print("self.kitting_base_y-----Pose2Robot",self.kitting_base_y)
        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[pose.position.x,pose.position.y,pose.position.z]
        world_target=self.relative_coordinate(ref_coord,curr_coord)
        print("ref_coord",ref_coord)
        print("world_target",world_target)
        print("curr_coord",curr_coord)
        base_target = Pose()
        base_target.position.x=world_target[0]
        base_target.position.y=world_target[1]
        base_target.position.z=world_target[2]
        # base_target.orientation = ee_target_tf.transform.rotation
        base_target.orientation = pose.orientation
        # #print base_target.orientation
        target = Pose2Matrix(base_target)
        
        return target

    def relative_coordinate(self,ref_coord, curr_coord):
        ref_coord = np.array(ref_coord)
        curr_coord = np.array(curr_coord)

        rel_coord = curr_coord - ref_coord
        return rel_coord


def main(args=None):
    rclpy.init(args=args)

    bandit_action_server = FloorActionServer()
    
    mt_executor = MultiThreadedExecutor(num_threads=7)
    mt_executor.add_node(bandit_action_server)
    mt_executor.spin()

    bandit_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()