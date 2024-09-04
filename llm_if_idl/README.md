# llm_if_idl
Action and Message interface for the ROS 2 Large Language Model InerFace.

TODO
Perform inference using clients from:
 - cortex (link)

### debug

′′′bash
colcon build --packages-select llm_if_idl
ros2 interface list | grep llm_if_idl
ros2 interface show llm_if_idl/action/ChatCompletion
ros2 interface show llm_if_idl/msg/ChatMessage
′′′