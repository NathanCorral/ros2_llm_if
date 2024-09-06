#include "rclcpp/rclcpp.hpp"

#include "llm_if_server/cortex_chat_completion_node.hpp"

namespace llm_if {
class CortexInterfaceComponent {
public:
  CortexInterfaceComponent(const rclcpp::NodeOptions &options)
      : node_ptr_(rclcpp::Node::make_shared("cortex_chat_completion", options)), cortex_interface_node_(node_ptr_) {};

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr get_node_base_interface() const {
    return node_ptr_->get_node_base_interface();
  }

protected:
  rclcpp::Node::SharedPtr node_ptr_;
  llm_if::CortexInterfaceNode cortex_interface_node_;
};
} // end of namespace whisper
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(llm_if::CortexInterfaceComponent)