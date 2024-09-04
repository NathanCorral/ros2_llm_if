cmake_minimum_required(VERSION 3.8)
project(llm_if_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp_components REQUIRED)


set(LLM_IF_NODES_DEPENDENCIES
  builtin_interfaces
  rcl_interfaces
  rclcpp
  rclcpp_action
  std_msgs
  std_srvs
  llm_if_idl
)


foreach(DEPENDENCY IN LISTS LLM_IF_NODES_DEPENDENCIES)
  find_package(${DEPENDENCY} REQUIRED)  
endforeach()

# llm interface nodes
add_library(${PROJECT_NAME} SHARED
  src/cortex_chat_completion_node.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  ${LLM_IF_NODES_DEPENDENCIES}
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ${LLM_IF_NODES_DEPENDENCIES}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

# Component todo #
# add_library(cortex_service_component SHARED
#   src/cortex_service_component.cpp
# )

# ament_target_dependencies(cortex_service_component
#   rclcpp
#   rclcpp_components
# )

# target_link_libraries(cortex_service_component
#   ${PROJECT_NAME}
# )

# rclcpp_components_register_node(cortex_service_component
#   PLUGIN cortex_service::InferenceComponent
#   EXECUTABLE cortex_service
# )

# # install components
# install(
#   TARGETS cortex_service_component
#   LIBRARY DESTINATION lib
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# install configurations
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
