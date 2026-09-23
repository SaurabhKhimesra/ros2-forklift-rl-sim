cmake_minimum_required(VERSION 3.8)
project(ros_gazebo_plugins)

# Builds libros_collision_detection.so, a Gazebo sensor plugin that republishes
# per-link contact state on /collision_detection/<link_name>.
#
# The install destination below has to agree with the <gazebo_ros
# plugin_path="${prefix}/../../lib"/> export in package.xml, or Gazebo will not
# find the library without GAZEBO_PLUGIN_PATH being set by hand.

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

add_library(ros_collision_detection SHARED src/ros_collision_detection_plugin.cpp)
ament_target_dependencies(ros_collision_detection gazebo gazebo_msgs gazebo_ros rclcpp)
target_include_directories(ros_collision_detection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(ros_collision_detection PUBLIC c_std_99 cxx_std_17)

# lib/, not lib/${PROJECT_NAME}: this must match the plugin_path exported in
# package.xml, which is how Gazebo discovers the plugin.
install(TARGETS ros_collision_detection LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
