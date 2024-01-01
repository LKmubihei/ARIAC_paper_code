#!/usr/bin/env python3

from pddl.logic import Predicate, constants, variables
from pddl.core import Domain, Problem, Action, Requirements
from pddl.formatter import domain_to_string, problem_to_string

# set up variables and constants
[robot] = variables("robot", types=["robot"])
[part] = variables("part", types=["part"])
[agv] = variables("AGV", types=["AGV"])
[container,source_container,destination_container]= variables("container source_container destination_container", types=["container"])
[assemblystation] = variables("assemblystation", types=["assemblystation"])


# define predicates 
on = Predicate("on", part, container)
flip_on = Predicate("flip_on", part, container)
put_on = Predicate("put_on", part, agv)
as_on = Predicate("as_on", part, assemblystation)
is_enabled = Predicate("is_enabled", robot)
is_reachable = Predicate("is_reachable", robot, container)
is_reachable_as = Predicate("is_reachable_as", robot, assemblystation)
can_arrive_at = Predicate("can_arrive_at", agv, assemblystation)
attach= Predicate("attach", part, robot)

# 当物料 part 位于容器 container 上、机器人 robot 启用且能够到达容器时，机器人可以将物料放置在AGV agv 上。
grasp = Action(
    "grasp",
    parameters=[robot, part, container],
    precondition=(on(part, container) or flip_on(part, container)) & is_enabled(robot) & is_reachable(robot, container)& (~attach(part, robot)),
    effect=attach(part, robot)
)

move = Action(
    "move",
    parameters=[robot, source_container, destination_container],
    precondition=is_enabled(robot) & is_reachable(robot, source_container),
    effect=(~is_reachable(robot, source_container) & is_reachable(robot, destination_container))  # 机器人现在在目标容器上
)


place = Action(
    "place",
    parameters=[robot, part, container],
    precondition=is_enabled(robot) & is_reachable(robot, container)&attach(part, robot),
    effect=(~attach(part, robot) & on(part, container))  # 机器人现在在目标容器上
)

flip = Action(
    "flip",
    parameters=[robot, part, container],
    precondition=is_enabled(robot) & is_reachable(robot, container)&flip_on(part, container),
    effect= on(part, container) # 机器人现在在目标容器上
)

# define Domain
requirements = [Requirements.STRIPS, Requirements.TYPING]
domain = Domain("ariac_domain",
                requirements=requirements,
                types={"robot": None,"part": None,"container": None,"assemblystation":None},
                predicates=[on, is_enabled, is_reachable,attach],
                actions=[grasp, move,place,flip])

floor_robot,ceiling_robot = constants("floor_robot ceiling_robot", types=["robot"])
bin1, bin2, bin5, bin6 ,curr_position,agv1,agv2,agv3,agv4= constants("bin1 bin2 bin5 bin6 curr_position agv1 agv2 agv3 agv4", types=["container"])
bin3, bin4, bin7, bin8= constants("bin3 bin4 bin7 bin8", types=["container"])
battery_red, pump_red, regulator_red, sensor_red = constants("battery_red pump_red regulator_red sensor_red", types=["part"])
battery_green, pump_green, regulator_green, sensor_green = constants("battery_green pump_green regulator_green sensor_green", types=["part"])
battery_orange, pump_orange, regulator_orange, sensor_orange = constants("battery_orange pump_orange regulator_orange sensor_orange ", types=["part"])
battery_blue, pump_blue, regulator_blue, sensor_blue = constants("battery_blue pump_blue regulator_blue sensor_blue", types=["part"])
battery_purple, pump_purple, regulator_purple, sensor_purple = constants("battery_purple pump_purple regulator_purple sensor_purple", types=["part"])
pddl_part_list={
    'battery_red': battery_red,
    'pump_red': pump_red,
    'regulator_red': regulator_red,
    'sensor_red': sensor_red,
    'battery_green': battery_green,
    'pump_green': pump_green,
    'regulator_green': regulator_green,
    'sensor_green': sensor_green,
    'battery_orange': battery_orange,
    'pump_orange': pump_orange,
    'regulator_orange': regulator_orange,
    'sensor_orange': sensor_orange,
    'battery_blue': battery_blue,
    'pump_blue': pump_blue,
    'regulator_blue': regulator_blue,
    'sensor_blue': sensor_blue,
    'battery_purple': battery_purple,
    'pump_purple': pump_purple,
    'regulator_purple': regulator_purple,
    'sensor_purple': sensor_purple
}

pddl_floor_position={'bin1':bin1,'bin2':bin2,'bin3':bin3,'bin4':bin4,
                'bin5':bin5,'bin6':bin6,'bin7':bin7,'bin8':bin8,
                'curr_position':curr_position}
# print("floor_robot",floor_robot.)

problem = Problem(
    "ariac_problem",
    domain=domain,
    requirements=requirements,
    objects=[ceiling_robot,bin1,bin2,bin5,bin6,agv1,agv2,agv3,agv4,curr_position,bin3, bin4, bin7, bin8,
             battery_red, pump_red, regulator_red, sensor_red ,battery_green, pump_green, regulator_green, sensor_green ,
             battery_orange, pump_orange, regulator_orange, sensor_orange,battery_blue, pump_blue, regulator_blue, sensor_blue ,
             battery_purple, pump_purple, regulator_purple, sensor_purple ],
    init=[on(battery_red,bin1),on(pump_red,bin1),on(regulator_green,bin2),on(sensor_green, bin2),
          is_reachable(ceiling_robot, curr_position),is_enabled(ceiling_robot)],
    goal=on(battery_red, agv1)
)

# print(problem_to_string(problem))

# 打开文件
fo = open("ariac_task_domain.pddl", "w")
str1 = domain_to_string(domain)
fo.write(str1)
fo.close()

# 打开文件
fo = open("ariac_task_problem.pddl", "w")
print("打印problem:",problem)
str1 = problem_to_string(problem)
fo.write(str1)
fo.close()

#################Test planner###########################

from pddl import parse_domain, parse_problem
from pddl_parser.planner import Planner
import sys
import time

# domain = parse_domain('paper_code/simple_1_domain.pddl')
# problem = parse_problem('paper_code/problem_1_problem.pddl')
domain='ariac_task_domain.pddl'
problem='ariac_task_problem.pddl'

start_time = time.time()
verbose = len(sys.argv) > 3 and sys.argv[3] == '-v'

planner = Planner()
plan = planner.solve(domain, problem)

print('Time: ' + str(time.time() - start_time) + 's')

if plan is not None:
    print('plan:')
    for act in plan:
        print(act if verbose else act.name + ' ' + ' '.join(act.parameters))
        # print(act)
else:
    sys.exit('No plan was found')