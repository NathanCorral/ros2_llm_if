# Minimum CMake version required
cmake_minimum_required(VERSION 3.8)

# Declare the project name
project(llm_if_server)

# Compiler-specific options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Add compilation flags if using GCC or Clang
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find the necessary dependencies for the project using ament
find_package(ament_cmake REQUIRED)
find_package(rclcpp_components REQUIRED)

# List of package dependencies used in the project
set(LLM_IF_NODES_DEPENDENCIES
  builtin_interfaces
  rcl_interfaces
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  CURL
  fmt
  llm_if_idl
)

# Loop over the dependencies and find each package
foreach(DEPENDENCY IN LISTS LLM_IF_NODES_DEPENDENCIES)
  find_package(${DEPENDENCY} REQUIRED)
endforeach()

# Include FetchContent for fetching external content
include(FetchContent)

# Fetch the nlohmann_json library from the specified URL
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)

# Make the fetched content available in the build
FetchContent_MakeAvailable(json)

# Uncomment to link the json library to the project
# target_link_libraries(${PROJECT_NAME}
#   nlohmann_json::nlohmann_json
# )

# Ensure that the json library doesn't get installed
set_target_properties(nlohmann_json PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Define a shared library for the project and specify source files
add_library(${PROJECT_NAME} SHARED
  src/cortex_chat_completion_node.cpp
  src/curl_manager.cpp  # Newly added source file
)

# Specify the dependencies for this target
ament_target_dependencies(${PROJECT_NAME}
  ${LLM_IF_NODES_DEPENDENCIES}
)

# Manually link the json library to the project
target_link_libraries(${PROJECT_NAME}
  nlohmann_json::nlohmann_json
)

# Include directories for the project (both during build and install)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Export the dependencies, but excluding nlohmann_json since it's excluded earlier
ament_export_dependencies(
  ${LLM_IF_NODES_DEPENDENCIES}
)

# Install the include directory (public headers)
install(
  DIRECTORY include/
  DESTINATION include
)

# Install the library and export it for use by other packages
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

# Component definitions
add_library(cortex_chat_completion_component SHARED
  src/cortex_chat_completion_component.cpp
)

# Specify component dependencies
ament_target_dependencies(cortex_chat_completion_component
  rclcpp
  rclcpp_components
)

# Link the main project and the json library to the component
target_link_libraries(cortex_chat_completion_component
  ${PROJECT_NAME}
  nlohmann_json::nlohmann_json
)

# Register the component with rclcpp (this sets it up as a ROS2 node)
rclcpp_components_register_node(cortex_chat_completion_component
  PLUGIN llm_if::CortexInterfaceComponent
  EXECUTABLE cortex_chat_completion
)

# Install the component as a shared library
install(
  TARGETS cortex_chat_completion_component
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install additional configuration files (e.g., launch, parameters)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# This marks the end of the package, indicating it’s ready to be built with ament
ament_package()
