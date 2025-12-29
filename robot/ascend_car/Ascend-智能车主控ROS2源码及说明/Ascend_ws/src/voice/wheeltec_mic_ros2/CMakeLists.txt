cmake_minimum_required(VERSION 3.8)
project(wheeltec_mic_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(
  include
)
#ament_index_register_resource("ros1_bridge_foreign_mapping")
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(wheeltec_mic_msg REQUIRED)
find_package(geometry_msgs REQUIRED) 

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(call_recognition src/call_recognition.cpp)
ament_target_dependencies(call_recognition rclcpp std_msgs wheeltec_mic_msg)
install(TARGETS call_recognition
  RUNTIME DESTINATION lib/${PROJECT_NAME}    # 程序运行的时候调用install中的路径
)
install(TARGETS
call_recognition
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(command_recognition src/command_recognition.cpp)
ament_target_dependencies(command_recognition rclcpp geometry_msgs std_msgs wheeltec_mic_msg)
install(TARGETS command_recognition
  RUNTIME DESTINATION lib/${PROJECT_NAME}    
)
install(TARGETS
command_recognition
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(node_feedback src/node_feedback.cpp)
ament_target_dependencies(node_feedback rclcpp std_msgs wheeltec_mic_msg)
install(TARGETS node_feedback
  RUNTIME DESTINATION lib/${PROJECT_NAME}   
)
install(TARGETS
node_feedback
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
