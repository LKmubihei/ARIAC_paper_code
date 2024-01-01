#!/usr/bin/env python3
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from std_msgs.msg import Float64,String
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import TransformException

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
import re

class CeilingActionServer(Node):
    gripper_states_ = {True: 'enabled',False: 'disabled'}

    def __init__(self):
        super().__init__('ceiling_action_server')

        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()
        self.group3 = MutuallyExclusiveCallbackGroup()
        self.group4 = MutuallyExclusiveCallbackGroup()   
        self.group5 = MutuallyExclusiveCallbackGroup()    
        self.group6 = MutuallyExclusiveCallbackGroup()  
        self.group7 = MutuallyExclusiveCallbackGroup()    


        self.Ceiling_robot_arm_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/ceiling_robot_controller/state', 
                                                self.ceiling_arm_joint_state_callback, qos_profile_sensor_data,callback_group=self.group1)
        self.gantry_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/gantry_controller/state',
                                                 self.gantry_state_callback, qos_profile_sensor_data,callback_group=self.group2)
        self.ceiling_action_client=ActionClient(self, FollowJointTrajectory, '/ceiling_robot_controller/follow_joint_trajectory',callback_group=self.group3)
        self.gantry_action_client=ActionClient(self, FollowJointTrajectory, '/gantry_controller/follow_joint_trajectory',callback_group=self.group4)
        self.ceiling_robot_gripper_state = VacuumGripperState()
        self.ceiling_gripper_enable = self.create_client(VacuumGripperControl, "/ariac/ceiling_robot_enable_gripper",callback_group=self.group5)
        self.ceiling_robot_gripper_state_sub = self.create_subscription(VacuumGripperState, 
                                                                      '/ariac/ceiling_robot_gripper_state', 
                                                                      self.ceiling_robot_gripper_state_cb, 
                                                                      qos_profile_sensor_data,callback_group=self.group6)
        self.ceiling_robot_tool_changer_ = self.create_client(ChangeGripper,'/ariac/ceiling_robot_change_gripper')   
        self.tree_action_server = ActionServer(self,Grasp,'grasp_2',self.execute_callback,callback_group=self.group7)
        self.get_logger().info('Action server created...')
        self.spin_lock = threading.Lock()

        # Setup TF listener
        self.tf_buffer = Buffer()
        self.tf_buffer_floor = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_listener_floor = TransformListener(self.tf_buffer_floor, self)
        self.goals = {}
        
        self.parts_complete =False
        self.ceiling_robot=False 
        
        self.ceiling_arm_joint_states=None
        
        self.ceiling_base_x = 0.0 
        self.ceiling_base_y = 0.0 
        self.ceiling_base_r = 0.0 

        self.pick_part_theta = 0.0 
        self.pick_torso_theta = 0.0 

        self.ceiling_torso_joint_names=[
            'gantry_x_axis_joint',  # x
            'gantry_rotation_joint', # r
            'gantry_y_axis_joint',  # y
            
           
        ]

        self.ceiling_torso_joints = None

        self.ceiling_arm_joint_names = [
            'ceiling_shoulder_pan_joint', 'ceiling_shoulder_lift_joint', 
            'ceiling_elbow_joint', 'ceiling_wrist_1_joint', 
            'ceiling_wrist_2_joint','ceiling_wrist_3_joint'
        ]
        self.ceiling_arm_normal_joints = None

        self.ceiling_init_position = [1.00,0.00, 0.00]

        self.ceiling_arm_init_position = [-3.179252150056005, -0.518062910410503, -1.9965621378670235, -0.630055723042239, 1.6116997462510554, -3.1415841107959963]

        self.init_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.017],
                                        [0.00,  -1.00, 0.00, -0.161],
                                        [0.00,  0.00, -1.00, 0.882],
                                        [0.00,  0.00, 0.00, 1.00]])

        self.ceiling_init_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                           [0.00, -1.00, 0.00],
                                           [0.00, 0.00, -1.00]]) 


        ##################gantry kitting姿态###############################
        self.ceiling_arm_kitting_position = [-0.019255781552382167, -2.11715423544646, 2.260508026291504,-0.14503290876189112, 1.5549932297936397, -3.141583271144321]

        self.ceiling_arm_kitting_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.343],
                                                    [0.00,  1.00, 0.00, 0.160],
                                                    [0.00,  0.00, 1.00, 0.453],
                                                    [0.00,  0.00, 0.00, 1.00]])

        self.ceiling_arm_kitting_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                                        [0.00, 1.00, 0.00],
                                                        [0.00, 0.00, 1.00]])
        self.rotate_rotation = numpy.matrix([[-1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  -1.00000000e+00,  0.00000000e+00],
                                        [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]])

        self.rotate_Y_90= numpy.matrix([[0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                        [0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00]])

        self.rotate_X_90= numpy.matrix([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  -1.00000000e+00],
                                        [0.00000000e+00, 1.00000000e+00,  0.00000000e+00]])

        self.flip_rotation = {
                'left_roll_0' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),
                'left_roll_pi' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),
                'right_roll_0' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),        
                'right_roll_pi' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),        
        }
        ##############gantry 装配姿态#################
        self.ceiling_arm_assemble_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                                        [0.00, -1.00, 0.00],
                                                        [0.00, 0.00, -1.00]])    
                                                         
        self.ceiling_arm_assemble_position = [-3.172421427905949, -1.664570820415968, -1.4746002176069055,0.3932066531825935, 1.6052786119583917, 0.000001]
        self.ceiling_arm_assemble_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.903],
                                                    [0.00,  -1.00, 0.00, -0.192],
                                                    [0.00,  0.00, -1.00, 0.562],
                                                    [0.00,  0.00, 0.00, 1.00]])    

                                                    
        self.ceiling_arm_flip_init_position = [-2.9508940961720214, -2.3241341588739353, -1.4187019275568016, -0.9695412285024076, 1.5707626896820814, -0.0000]   



    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        # Process the goal
        goal = goal_handle.request
        self.get_logger().info(f'Received goal: type={goal.type}, pose={goal.pose}')

        if goal.number==0:                                   # 0抓  1放 2移动 3翻转 
            self.ceiling_arm_init()  
            print("打印一下我收到的位置0",goal.pose.position)
            part=sPart(goal.type,goal.dest,goal.pose)
            self.pick_part_on_bin_ceiling(goal.dest,part)   
        elif goal.number==1:  
            part=sPart(goal.type,goal.dest,goal.pose)                                 # 
            self.ceiling_robot_place_part_on_kit_tray(part,goal.dest)
        elif goal.number==2:                                   # 
            self.move_to_ceiling(goal.dest)

        # elif goal.number==3:                                     # 
        #     self.flip_part(goal.type, goal.pose)


        # Perform some processing and set the result
        goal_handle.succeed()
        result = Grasp.Result()
        result.success = True
        print("对于",goal.number,"我已经发送:",result)

        return result



#region ## ###################################### Ceiling Robot  Function   ###############################################################################   
    
    def agv_to_as(self,subtask):
            if isinstance(subtask.agv_numbers, int):
                subtask.agv_numbers = [subtask.agv_numbers]

            for agv in subtask.agv_numbers:
                if subtask.station==1 or subtask.station==3:        # 由 station 推出 agv_destination
                    agv_destination=1
                if subtask.station==2 or subtask.station==4:
                    agv_destination=2  
                
                agv_location=[self.agv1_position,self.agv2_position,self.agv3_position,self.agv4_position]
                print("agv的location是:",agv_location)
                if agv_location[agv - 1] != agv_destination:
                    self.move_agv(agv, agv_destination)
                    self.lock_agv_tray(agv)

            sleep(0.5)

            if  not self.order_pose[subtask.order_id] :

                self.order_pose[subtask.order_id] =  self.get_assembly_poses(subtask.order_id)
                print("打印一下获得的pose:",self.order_pose[subtask.order_id])
            
 
    def ceiling_robot_gripper_state_cb(self, msg: VacuumGripperState):
        self.ceiling_robot_gripper_state = msg


    def set_ceiling_robot_gripper_state(self, state) -> bool: 
        
        if self.ceiling_robot_gripper_state.enabled == state:
            self.get_logger().warn(f'Gripper is already {self.gripper_states_[state]}')
            return
        
        request = VacuumGripperControl.Request()
        request.enable = state
        
        future = self.ceiling_gripper_enable.call_async(request)
        sleep(0.3)

        
        if future.result().success:
            self.get_logger().info(f'Changed gripper state to {self.gripper_states_[state]}')
        else:
            self.get_logger().warn('Unable to change gripper state')
        
        
    
    
    def gantry_state_callback(self, msg):
        self.ceiling_base_x = msg.actual.positions[0]
        self.ceiling_base_y = msg.actual.positions[1]
        self.ceiling_base_r = msg.actual.positions[2]

        self.ceiling_torso_joints = [self.ceiling_base_x,self.ceiling_base_r,self.ceiling_base_y]
        
    def ceiling_arm_joint_state_callback(self,msg):
        self.ceiling_arm_joint_states = msg.actual.positions


    
    def gantry_robot_init(self):
        while not self.ceiling_arm_joint_states and rclpy.ok():
            sleep(0.5)
            #print 'waiting for initialization ...'
        # 复位
        print ("gantry_robot_initializing...")
        self.MOV_A_CEILING(self.ceiling_arm_init_position, eps=0.01,sleep=False)

        
        eps = 0.02
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            # q_begin = copy.deepcopy(self.ceiling_torso_joints)
            # print("q_began",q_begin)
            q_end = self.ceiling_init_position
            # print("q_end",q_end)
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            distance = max(delta)
            # print("distance",distance)
            if distance < eps:
                break
            move_time = distance/gantry_velocity
            # print("move_time",move_time)
            # print("-------")
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)

        print ("gantry_robot_initialize success!")
        return True
    
    def send_gantry_to_state(self,positions,time_from_start):
        msg = JointTrajectory()
        msg.joint_names = self.ceiling_torso_joint_names
        point = JointTrajectoryPoint()
        tran_position = [positions[0], positions[1],-positions[2]]
        point.positions = tran_position
        point.time_from_start = Duration(seconds=time_from_start).to_msg()
        msg.points = [point] 
        
        self.move(self.gantry_action_client,msg)
        

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
            self.ceiling_robot_goal_response_callback)

    def ceiling_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._ceiling_robot_get_result_future = goal_handle.get_result_async()



    def MOV_A_CEILING(self, target_joint_angles,time_from_start = 0.0,eps = 0.01,sleep = True):
        '''
        6-DOF，角度顺序是 [elbow,...]
        time_from_start,默认按照最快时间执行
        eps = 0.0025
        '''
        #当前关节角度
        q_begin = self.ceiling_arm_joint_states
        # q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
        q_end = target_joint_angles
        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        distance = max(delta)
        # 计算理想的运行时间
        runtime = distance/gantry_angle_velocity
        exe_time = 0.00
        start_time = self.get_clock().now().nanoseconds/1e9
        sleep_flag = True


        while rclpy.ok() and sleep_flag and exe_time < runtime*2:
            exe_time = self.get_clock().now().nanoseconds/1e9 - start_time
            #当前关节角度
            q_begin = self.ceiling_arm_joint_states
            # q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
            q_end = target_joint_angles
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            distance = max(delta)
            # #print delta
            if distance < eps:
                break
            # 计算理想的运行时间
            if time_from_start == 0.0:
                time_from_start = distance/gantry_angle_velocity
            # #print time_from_start
            #做运动插补
            traj = traj_generate(self.ceiling_arm_joint_names,q_begin,q_end,time_from_start)
            ##print traj
            self.move(self.ceiling_action_client,traj)
            
            if sleep:
                time.sleep(time_from_start)
            else:
                sleep_flag = False   
                
    def move_to_ceiling(self, location, right=0.0,left=0.0,forward=0.0,back=0.0,flip=False):
        '''
        location 表示驻点
        '''
        # print("开始移动") 
        # self.robot_info.next_park_location = location
        # if not flip:
        #     self.robot_info.work_state= "moving"
        # else:
        #     self.robot_info.work_state= "flipping"

        if "conveyor" in location:
            target_torso_point  = [0.3+5, -pi/2, self.ceiling_base_y]
        else:
            end_point = gantry_robot_park_location[location]
            target_torso_point = [end_point[0]+forward-back,end_point[2],end_point[1]+right-left]


        distance = max([abs(target_torso_point[0]-self.ceiling_base_x), abs(target_torso_point[1]- self.ceiling_base_r)\
            ,abs(-target_torso_point[2]-self.ceiling_base_y)])

        time_from_start = distance/gantry_velocity
        print("开始移动")
        self.send_gantry_to_state(target_torso_point,time_from_start)
        time.sleep(time_from_start)
        print("target_torso_point",target_torso_point)
        print("self.ceiling_torso_joints",self.ceiling_torso_joints)
        print("--------")
        print("已到达")   

    def Left(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[2] = q_end[2]-distance

        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        dis = max(delta)
        move_time = dis/gantry_velocity
        self.send_gantry_to_state(q_end, move_time)
        sleep(move_time)

    def Right(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[2] = q_end[2]+distance

        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        dis = max(delta)
        move_time = dis/gantry_velocity
        self.send_gantry_to_state(q_end, move_time)
        sleep(move_time)
        return True

    def Back(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[0] = q_end[0]-distance
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            dis = max(delta)
            if dis < eps:
                break
            move_time = dis/gantry_velocity
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)

        return True

    def Forward(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[0] = q_end[0]+distance
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            dis = max(delta)
            if dis < eps:
                break
            move_time = dis/gantry_velocity
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)
        return True


    def pick_part_on_bin_ceiling(self,location,part,distance = 0.5,repick_callback_num = 0):

        target_matrix =self.Tf_trans(part.pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]

        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        self.set_ceiling_robot_gripper_state(True)
        
        if 'battery' in part.type:
            need_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]-0.025
            
            print("在求证呢need_yaw:",need_yaw,"position:",position)
            
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        
        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 3:
                self.robot_arm_init(location, part)
                return False
                
            repick_nums = repick_nums + 1
            p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height +0.07- (repick_nums-1)*0.003
            p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.15
        
            self.MOV_M_ceiling(p1,eps =0.01,time_factor=3)
            print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

            self.MOV_M_ceiling(p2,eps =0.01)
            
        p1[0,3]=p1[0,3]+0.4
        self.MOV_M_ceiling(p1,eps =0.01)
        self.robot_arm_init(location,part)
 
    def pick_part_has_flip_ceiling(self,location,part,agv_number,distance = 0.5,repick_callback_num = 0):

        target_bin=find_closest_bin(agv_number)
        target_part_list_bins = self.search_part_on_bins(part.type)
        
        if not target_part_list_bins:
            return False
            
        target_part=find_nearest_part(target_part_list_bins,bin_position[target_bin][0],bin_position[target_bin][1]+0.3)
        
        

        
        target_matrix =self.Tf_trans(target_part.pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]


        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        
        if 'battery' in part.type:
            position[2]=position[2]-0.03
        #     position[0]=position[0]-0.068
        # if 'pump' in part.type:
        #     position[1]=position[1]-0.03
        #     position[0]=position[0]-0.12
        # if 'sensor' in part.type:
        #     position[0]=position[0]-0.07
        # if 'regulator' in part.type:
        #     position[1]=position[1]+0.03
        #     position[0]=position[0]-0.07

            
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        self.set_ceiling_robot_gripper_state(True)
        


        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 3:
                self.robot_arm_init(location, part)
                return False
                
            repick_nums = repick_nums + 1
            p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height +0.07- (repick_nums-1)*0.003
            p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.15
        
            self.MOV_M_ceiling(p1,eps =0.01)
            print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

            self.MOV_M_ceiling(p2,eps =0.01)
            
        p1[0,3]=p1[0,3]+0.2
        self.MOV_M_ceiling(p1,eps =0.01)
        self.robot_arm_init(location,part)
    
        return target_part


    def pick_part_flip__on_bin_ceiling(self,location,part,distance = 0.5,repick_callback_num = 0):

        target_matrix =self.Tf_trans(part.pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]

        if 'battery' in part.type:
            
            need_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]+0.005

        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  
        

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        self.set_ceiling_robot_gripper_state(True)
        repick_nums = 0
        
        print("来执行翻转任务了")
        if "pump" in part.type:
            while not self.ceiling_robot_gripper_state.attached:
                if repick_nums >= 3:
                    self.robot_arm_init(location, part)
                    return False
                    
                repick_nums = repick_nums + 1
                p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height -0.049- (repick_nums-1)*0.001
                p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.08
            
                self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
                print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

                self.MOV_M_ceiling(p2,eps =0.01,time_factor=5)
        
        else:
            while not self.ceiling_robot_gripper_state.attached:
                if repick_nums >= 3:
                    self.robot_arm_init(location, part)
                    return False
                    
                repick_nums = repick_nums + 1
                p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height -0.00- (repick_nums-1)*0.001
                p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.08
            
                self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
                print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

                self.MOV_M_ceiling(p2,eps =0.01,time_factor=5)
            
        p1[0,3]=p1[0,3]+0.3
        self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
        self.robot_arm_init(location,part)


    def flip_part_on_ceiling(self,agv_number,part):    
        
        target_bin=find_closest_bin_ceiling(agv_number)
        self.move_to_ceiling(target_bin)

        part_pose = Pose()             
        result = bin3_8_flip[target_bin]

        part_pose.position.x = result[0]
        part_pose.position.y = result[1]
        part_pose.position.z = result[2]
        part_pose.orientation.x = 0.0
        part_pose.orientation.y = 0.0
        part_pose.orientation.z = 0.0
        part_pose.orientation.w = 1.0 
        
        target_matrix = self.Tf_trans(part_pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2
        self.MOV_M_ceiling(p2)
        p2[0,3]=p2[0,3]-0.06
        self.MOV_M_ceiling(p2,time_factor=3)

        self.set_ceiling_robot_gripper_state(False)
        

        p3 = Rot2Matrix(self.flip_rotation['right_roll_0'], position)
        p5=copy.deepcopy(p3)
        p5[0,3]=p5[0,3]+0.2
        p5[1,3]=p5[1,3]+0.1
        self.MOV_M_ceiling(p5,eps =0.01)  
        self.set_ceiling_robot_gripper_state(True)
        
        if 'pump' in part.type:

            p5[0,3]=p5[0,3]-0.12
            p5[1,3]=p5[1,3]-0.03
            self.MOV_M_ceiling(p5,eps =0.01,time_factor=5)  

            pick_num=0
            while not self.ceiling_robot_gripper_state.attached:
                if pick_num>10:
                    break
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001
                self.MOV_M_ceiling(p5,eps =0.001,time_factor=20) 
                print(f"第{pick_num}次抓取")
                sleep(0.5) 
            
            p5[0,3]=p5[0,3]+0.2
            self.MOV_M_ceiling(p5,eps =0.01)   
            self.ceiling_arm_init()  
        





    def ceiling_arm_init(self):
        self.MOV_A_CEILING(self.ceiling_arm_init_position, eps=0.02)
        # self.robot_info.work_state= "standby"
        
    def robot_arm_init(self, target_agv_tray, target_part):
        # self.gripper.deactivate_gripper()

        if 'ks' in target_agv_tray and target_part.pose.position.x < -2.2828:
            p3 = self.ceiling_arm_kitting_position
            self.MOV_A_CEILING(p3,eps =0.02) 
            if "agv4" in target_agv_tray :
                self.MOV_A_CEILING(self.ceiling_arm_init_position,eps =0.05) 
            else:
                pass
        else:
            p3 = self.ceiling_arm_init_position
            self.MOV_A_CEILING(p3,eps =0.02) 
    
    
        
    def send_gantry_arm_to_state(self,positions,time_from_start):
        msg = JointTrajectory()
        msg.joint_names = self.ceiling_arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=time_from_start).to_msg()
        msg.points = [point]
        #rospy.loginfo("Sending command:\n" + str(msg))
        self.move(self.ceiling_action_client,msg)
    


    def STOP_ceiling(self):
        self.MOV_A_CEILING(self.ceiling_arm_joint_states)
        # self.send_gantry_to_state(self.ceiling_torso_joints,0.001)

    def MOV_M_ceiling(self, target_matrix,time_from_start = 0.0,eps = 0.005,sleep = True,flip_flag=False,time_factor=1.0): 
        '''
        4*4 Matrix 
        主要用来做抓取,eps = 0.0025是机器人的最小误差，一般设置要大于此值
        '''
        q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
        print("target_matrix",target_matrix)
        target = IKinematic(target_matrix,q_begin,A_k)
        if target==None:
            target = IKinematic(target_matrix,q_begin,A_k)
            if target==None:
                print("解不出来")
                return False
        # #print "IKinematic_result:",target
        if flip_flag:
            target[-1] = q_begin[-1]
        #求时间
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        ##print delta
        distance = max(delta)
        time_from_start = distance/gantry_angle_velocity*time_factor
        traj = traj_generate(self.ceiling_arm_joint_names,q_begin,target,time_from_start)
        self.move(self.ceiling_action_client,traj)
        time.sleep(time_from_start)
        return True
        


    def Tf_trans(self, pose):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        ''' 
        sleep(0.5)   #机器人停稳了
        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'gantry_target_frame'
        translation=Vector3()
        translation.x=pose.position.x
        translation.y=pose.position.y
        translation.z=pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        sleep(0.5)
        
        t =TransformStamped()

        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", "gantry_target_frame",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
          

        print("-----t----",t) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation
        print("-----pose----",pose)
        target = Pose2Matrix(pose)
        return target
    
    def Tf_trans_pose(self, pose):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        ''' 
        sleep(0.6)   #机器人停稳了
        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'assembly_part_frame'
        translation=Vector3()
        translation.x=pose.position.x
        translation.y=pose.position.y
        translation.z=pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        sleep(0.5)
        
        
        t =TransformStamped()
        t1 =TransformStamped()
        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", "assembly_part_frame",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')           

        print("-----t1----",t1) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation

        target = Pose2Matrix(pose)
        return target

    def FrameWorldPose_CEILING(self,frame_id):
        t =TransformStamped()
        pose = Pose()
        
        MAX_ATTEMPTS = 20
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", frame_id,  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
                return

        print("-----与AGV的变换----",t)
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        target = Pose2Matrix(pose)

        try:
            t1= self.tf_buffer.lookup_transform("world", frame_id,  rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
            return
        print("-----我想看看{frame_id}在world中的坐标----",t1)
        return target
        
    
    

    def goal_response_callback(self, future):
        goal_handle = future.result()
        goal_trajectory = self.goals.pop(future)
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected')
            return
        self.get_logger().info(f'Goal  accepted')


    def ceiling_robot_place_part_on_kit_tray(self,target_part,dest) -> bool:
        
        print("-----我想看看目标位置{dest}----",dest)
        numbers = re.findall('agv(\d+)', dest)
        agv_number = int(numbers[0])
        part_pose = Pose()             
        result = agv_quadrant_position(agv_number, 1)

        part_pose.position.x = result[0]
        part_pose.position.y = result[1]
        part_pose.position.z = result[2]
        part_pose.orientation.x = 0.0
        part_pose.orientation.y = 0.0
        part_pose.orientation.z = 0.0
        part_pose.orientation.w = 1.0 
        
        # print("我要去的是象限",subtask.product_quadrant)

        target_matrix = self.Tf_trans(part_pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2
        
        self.MOV_M_ceiling(p2)

        self.set_ceiling_robot_gripper_state(False)

        self.ceiling_arm_init()
        
        # self.set_ceiling_robot_gripper_state(True)
    
        # print("开始抓取坏零件")
        # p2 = copy.deepcopy(target_matrix)
        # self.move_to_ceiling(dest,forward=0.2)
        # p2[2,3]=p2[2,3]-0.2
        # repick_nums = 0

        # while not self.ceiling_robot_gripper_state.attached:
        #     if repick_nums >= 10:
        #         self.ceiling_arm_init()
        #         return False
        #     print(f"第{repick_nums}次抓取坏零件")
        #     repick_nums = repick_nums + 1
        #     p2[0,3] =target_matrix[0,3]+ceiling_faulty_part[target_part.type]-0.003*repick_nums
        #     # p2[0,3]=p2[0,3]- 0.01*repick_nums
        #     self.MOV_M_ceiling(p2)
        #     sleep(0.2)
            
        # self.ceiling_arm_init()  
        # self.move_to_ceiling("can")    
        # self.set_ceiling_robot_gripper_state(False)
 
        # return True 

    def ceiling_pick_assembly_part(self,subtask) -> bool:
        self.ceiling_arm_init()
        target_matrix = self.Tf_trans_pose(subtask.pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        if 'battery' in subtask.type:
            need_yaw=GetYaw(subtask.pose.orientation)-1.574971916153666
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]-0.027
        if 'pump' in subtask.type:
            position[0]=position[0]+0.003

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2

        self.MOV_M_ceiling(p2)

        self.set_ceiling_robot_gripper_state(True)
        p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[subtask.type]+vacuum_gripper_height +0.07
        
        self.MOV_M_ceiling(p1,eps =0.005,time_factor=5)
        sleep(0.5)

        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 5:
                print("抓取失败")
                return False
            
            print(f"这是抓取第{repick_nums}次")
            repick_nums = repick_nums + 1
            p1[0,3] = p1[0,3]- 0.002
            self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
            sleep(0.5)

        # 
        print("self.attached",self.ceiling_robot_gripper_state.enabled)
        if self.ceiling_robot_gripper_state.attached:
            self.ceiling_robot_info.work_state = "has_grasped"

        
        p1[0,3]=p1[0,3]+0.5
        self.MOV_M_ceiling(p1,eps =0.01)
        
        return True 
    
    def TF_trans_two(self,as_station,pose):
        sleep(0.5)   #机器人停稳了
        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = as_station
        tf_msg.child_frame_id = 'put_assembly_part'
        translation=Vector3()
        translation.x=pose.pose.position.x
        translation.y=pose.pose.position.y
        translation.z=pose.pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        
        sleep(0.5)
        
        
        t =TransformStamped()
        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", "put_assembly_part",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')           

        print("-----t----",t) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation
        print("-----pose----",pose)
        target = Pose2Matrix(pose)
        return target                     

    def ceiling_place_assembly_pump(self,subtask,cmd_type,part) -> bool:
        self.move_to_ceiling("as"+str(subtask.station))  
        self.ceiling_arm_init()
        sleep(0.5)
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_rotation, position)
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            yaw=GetYaw(subtask.assembled_pose.pose.orientation)
            if yaw>0:
                base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            else:
                base_target.orientation=QuaternionFromRPY(need_yaw+3.1415826,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            if part_yaw>0:
                base_target.orientation=QuaternionFromRPY(part_yaw,0,0)     #让夹爪旋转
            else:
                base_target.orientation=QuaternionFromRPY(part_yaw+3.1415826,0,0)     #让夹爪旋转

            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        p1 = copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]+0.3
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.15
        p2[2,3]=p2[2,3]-0.001
 
        
        self.MOV_M_ceiling(p1)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
            
        sleep(0.2)
        self.MOV_M_ceiling(p2,time_factor=5)
        
        # self.MOV_M_ceiling(p1,time_factor=10)
        
        
    def ceiling_place_assembly_sensor(self,subtask,cmd_type,part) -> bool:
        
        self.move_to_ceiling("as"+str(subtask.station),right=0.1)  
        self.ceiling_arm_init()
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix1 = Rot2Matrix(self.rotate_Y_90, position)
        
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666        
            base_target.orientation=QuaternionFromRPY(need_yaw,1.574971916153666,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
        
        else:
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))     
            base_target.orientation=QuaternionFromRPY(part_yaw,1.574971916153666,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
        

        target_matrix1[0,3]=target_matrix1[0,3]+0.1
        
        p1 = copy.deepcopy(target_matrix1)
        p1[0,3]=p1[0,3]+0.3

        p1=p1@part_to_gripper  
        self.MOV_M_ceiling(p1,time_factor=3)
        sleep(0.2)

        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
 
        p3=copy.deepcopy(p1)
        p3[1,3]=p3[1,3]+0.050
        # p3[2,3]=p3[2,3]+0.045
        p3[0,3]=p3[0,3]-0.319
        self.MOV_M_ceiling(p3,time_factor=5)
        
        sleep(0.2)

        p3[1,3]=p3[1,3]-0.050
        self.MOV_M_ceiling(p3,time_factor=10)

    def ceiling_place_assembly_regulator(self,subtask,cmd_type,part) -> bool:
        #---------------------------------------------------------------------------------------------------------
        self.move_to_ceiling("as"+str(subtask.station))  
        self.ceiling_arm_init()
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_Y_90, position)      
        if cmd_type=="assembly":  
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0     
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            

            base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            
            if part.need_flip:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            else:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
                print("part_yaw:::::",part_yaw)
            base_target.orientation=QuaternionFromRPY(part_yaw,0,0) 

            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper
            
        target_matrix[0,3]=target_matrix[0,3]+0.1
        self.MOV_M_ceiling(target_matrix,time_factor=2)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
        
        p1=copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]-0.07
        p1[2,3]=p1[2,3]+0.075
        sleep(0.2)
        self.MOV_M_ceiling(p1,time_factor=3)

        sleep(0.2)
        p1[0,3]=p1[0,3]-0.025
        self.MOV_M_ceiling(p1,time_factor=5)


   
    def ceiling_place_assembly_battery(self,subtask,cmd_type,part) -> bool:
        
        self.ceiling_arm_init()
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_rotation, position)
        
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper    
        
        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0

            if part.need_flip:
                part_yaw=1.574971916153666
            else:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            base_target.orientation=QuaternionFromRPY(part_yaw,0,0) 
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper     
        
        target_matrix[0,3]=target_matrix[0,3]+0.3
        self.MOV_M_ceiling(target_matrix)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
            

        p1=copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]-0.2
        p1[1,3]=p1[1,3]-0.095
        self.MOV_M_ceiling(p1,time_factor=3)
        sleep(0.2)

        
        p1[0,3]=p1[0,3]-0.045
        self.MOV_M_ceiling(p1,time_factor=5)
        sleep(0.2)
        
        p2=copy.deepcopy(p1)
        p2[1,3]=p2[1,3]+0.05
        self.MOV_M_ceiling(p2,time_factor=5)
        
#endregion 


def main(args=None):
    rclpy.init(args=args)

    bandit_action_server = CeilingActionServer()
    
    mt_executor = MultiThreadedExecutor(num_threads=7)
    mt_executor.add_node(bandit_action_server)
    mt_executor.spin()

    bandit_action_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()