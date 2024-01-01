cmake_minimum_required(VERSION 3.8)
project(paper_ws)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ariac_msgs REQUIRED)

find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
set(msg_files
"msg/Plan.msg"
)
set(action_files
"action/Grasp.action"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  DEPENDENCIES
  builtin_interfaces
  std_msgs
  geometry_msgs
  ADD_LINTER_TESTS
)



# Install Python modules    
ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION lib/${PROJECT_NAME})
ament_python_install_package(pddl_parser PACKAGE_DIR pddl_parser)
ament_python_install_package(behaviors PACKAGE_DIR behaviors)

# Install Python executables
install(PROGRAMS
  launch/start_competition.py          
  scripts/start_tree.py
  scripts/graph.py
  behaviors/execute_tree_1.py
  behaviors/floor_1_action_servers.py 
  behaviors/ceiling_action_servers.py 
  behaviors/floor_behaviors.py 
  paper_ws/paper.py
  paper_ws/robot_planner.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY 
  config
  launch
  scripts
  pddl_parser
  behaviors
  action
  msg
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
