#ifndef LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_
#define LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_

#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
// #include "std_msgs/msg/int16_multi_array.hpp"

#include "llm_if_idl/action/chat_completion.hpp"
// #include "whisper_util/audio_buffers.hpp"
// #include "whisper_util/model_manager.hpp"
// #include "whisper_util/whisper.hpp"

namespace llm_if {
class CortexInterfaceNode {
  using ChatCompletion = llm_if_idl::action::ChatCompletion;
  using GoalHandleChatCompletion = rclcpp_action::ServerGoalHandle<ChatCompletion>;

public:
  CortexInterfaceNode(const rclcpp::Node::SharedPtr node_ptr);

protected:
  rclcpp::Node::SharedPtr node_ptr_;

  // parameters
  void declare_parameters_();
  // rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr on_parameter_set_handle_;
  // rcl_interfaces::msg::SetParametersResult
  // on_parameter_set_(const std::vector<rclcpp::Parameter> &parameters);

  // No subscriptions
  // rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr audio_sub_;
  // void on_audio_(const std_msgs::msg::Int16MultiArray::SharedPtr msg);

  // action server
  rclcpp_action::Server<ChatCompletion>::SharedPtr inference_action_server_;
  rclcpp_action::GoalResponse on_inference_(const rclcpp_action::GoalUUID &uuid,
                                            std::shared_ptr<const ChatCompletion::Goal> goal);
  rclcpp_action::CancelResponse
  on_cancel_inference_(const std::shared_ptr<GoalHandleChatCompletion> goal_handle);
  void on_inference_accepted_(const std::shared_ptr<GoalHandleChatCompletion> goal_handle);
  // std::string inference_(const std::vector<float> &);
  rclcpp::Time chat_completion_start_time_;

  // llm_if data
  // std::unique_ptr<ModelManager> model_manager_;
  // std::unique_ptr<BatchedBuffer> batched_buffer_;
  // std::unique_ptr<Whisper> whisper_;
  // std::string language_;
  // void initialize_cortex_();
  // void initialize_model_();
};
} // end of namespace llm_if
#endif // LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_