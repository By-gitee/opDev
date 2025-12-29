cmake_minimum_required(VERSION 3.5)
project(voice_control)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(serial REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(wheeltec_mic_msg REQUIRED)
#find_package(ecl_threads REQUIRED)


link_directories(
  lib/x64
)

find_package(rosidl_default_generators REQUIRED)


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

ament_package()

include_directories(
  ./include
)


add_executable(voicecontrol src/voice_control.cpp)
ament_target_dependencies(voicecontrol rclcpp std_msgs wheeltec_mic_msg)
target_link_libraries(voicecontrol serial offline_record_lib hid_lib)
#target_link_libraries(voicecontrol geometry_msgs)
ament_target_dependencies(voicecontrol
  "rclcpp"
  "geometry_msgs"
  "serial"
  "sensor_msgs"
  "tf2"
  "nav_msgs"
  "wheeltec_mic_msg"
)

install(DIRECTORY launch config msg lib audio tmp feedback_voice
  DESTINATION share/${PROJECT_NAME}
)
install(TARGETS
voicecontrol
  DESTINATION lib/${PROJECT_NAME}
)

