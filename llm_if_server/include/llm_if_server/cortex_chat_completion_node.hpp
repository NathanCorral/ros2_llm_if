#ifndef LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_
#define LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_

#include <chrono>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <fmt/core.h>  // Include fmt  // c++20 to remove

#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "llm_if_server/curl_manager.hpp"

#include "llm_if_idl/action/chat_completion.hpp"
#include "llm_if_idl/msg/chat_message.hpp"

namespace llm_if {
using json = nlohmann::json;

class CortexInterfaceNode {
  // Useful shorthands
  using ChatCompletion = llm_if_idl::action::ChatCompletion;
  using GoalHandleChatCompletion = rclcpp_action::ServerGoalHandle<ChatCompletion>;

public:
  CortexInterfaceNode(const rclcpp::Node::SharedPtr node_ptr);
  
protected:
  rclcpp::Node::SharedPtr node_ptr_;

  // ROS Action Server
  rclcpp_action::Server<ChatCompletion>::SharedPtr inference_action_server_;
  rclcpp_action::GoalResponse on_inference_(const rclcpp_action::GoalUUID& uuid,
                                            std::shared_ptr<const ChatCompletion::Goal> goal);
  rclcpp_action::CancelResponse
  on_cancel_inference_(const std::shared_ptr<GoalHandleChatCompletion> goal_handle);
  void on_inference_accepted_(const std::shared_ptr<GoalHandleChatCompletion> goal_handle);

private:
  // Tools for interfacing with curl API and cortex server
  CurlManager curl_manager_;

  // https://cortex.so/api-reference#tag/models/post/v1/models/{modelId}/start
  std::vector<std::string> models_started_;
  bool startModel(const std::string& model_id, std::string& response);

  // https://cortex.so/api-reference#tag/inference/post/v1/chat/completions
  rclcpp::Time chat_completion_start_time_;
  bool chatCompletion(const std::string& model_id, 
                      const json& chat_history, 
                      std::string& response,
                      const float& temperature = 0.8,
                      const bool& stream = false,
                      const int& max_tokens = 2048,
                      const float& top_p = 0.95);

  // Helper functions
  json formatChatHistory(const std::vector<llm_if_idl::msg::ChatMessage>& messages);
  /*
  Returns true if the model has already been loaded
  */
  bool modelLoaded(const std::string& model_id);

};
} // end of namespace llm_if
#endif // LLM_IF_NODES__CORTEX_CHAT_COMPLETION_NODE_HPP_