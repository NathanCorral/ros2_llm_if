#include "llm_if_server/cortex_chat_completion_node.hpp"

namespace llm_if {
CortexInterfaceNode::CortexInterfaceNode(const rclcpp::Node::SharedPtr node_ptr)
    : node_ptr_(node_ptr) 
{
  // inference action server
  inference_action_server_ = rclcpp_action::create_server<ChatCompletion>(
      node_ptr_, "cortex_chat_completion",
      std::bind(&CortexInterfaceNode::on_inference_, this, std::placeholders::_1, 
                                                            std::placeholders::_2),
      std::bind(&CortexInterfaceNode::on_cancel_inference_, this, std::placeholders::_1),
      std::bind(&CortexInterfaceNode::on_inference_accepted_, this, std::placeholders::_1));
}

rclcpp_action::GoalResponse
CortexInterfaceNode::on_inference_(const rclcpp_action::GoalUUID& /*uuid*/,
                             std::shared_ptr<const ChatCompletion::Goal> /*goal*/) 
{
  RCLCPP_INFO(node_ptr_->get_logger(), "Received inference request.");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
CortexInterfaceNode::on_cancel_inference_(
  const std::shared_ptr<GoalHandleChatCompletion> /*goal_handle*/) 
{
  RCLCPP_INFO(node_ptr_->get_logger(), "Cancelling inference...");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void CortexInterfaceNode::on_inference_accepted_(
    const std::shared_ptr<GoalHandleChatCompletion> goal_handle) 
{
  RCLCPP_INFO(node_ptr_->get_logger(), "Starting inference...");
  auto goal = goal_handle->get_goal();
  auto feedback = std::make_shared<ChatCompletion::Feedback>();
  auto result = std::make_shared<ChatCompletion::Result>();

  // Pull in data 
  auto model_id = goal->model;
  auto messages = goal->messages;
  auto temperature = goal->temperature;
  auto max_tokens = goal->max_tokens;
  auto top_p = goal->top_p;

  json chat_history = formatChatHistory(messages);

  // Check if the model has already been started, and start it on Cortex server if not
  feedback->status = "Loading model on Cortex Server...";
  if (!modelLoaded(model_id)) {
    std::string start_model_resp;
    if (!startModel(model_id, start_model_resp)) {
      result->info = fmt::format("Loading model failed:  {}", start_model_resp);
      RCLCPP_WARN(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->succeed(result);
      return;
    }
    feedback->status = fmt::format("Loaded model successful with resp: \"{}\"", start_model_resp);
  } else {
    feedback->status = fmt::format("Model \"{}\" already loaded", model_id);
  }

  // Check if cancelled
  if (goal_handle->is_canceling()) {
    result->info = "Inference cancelled.";
    RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
    goal_handle->canceled(result);
    return;
  }

  // Perform llm inference
  bool ret;
  std::string chat_resp;
  ret = chatCompletion(model_id, chat_history, chat_resp, temperature, false, max_tokens, top_p);

  if (!ret) {
    result->info = fmt::format("Chat Completion failed:  {}", chat_resp);
    RCLCPP_WARN(node_ptr_->get_logger(), result->info.c_str());
    goal_handle->succeed(result);
    return;
  }


  std::string info_msg = fmt::format("Chat History:  {}", chat_history.dump(2));
  RCLCPP_INFO(node_ptr_->get_logger(), info_msg.c_str());
  json resp = json::parse(chat_resp);
  info_msg = fmt::format("Response:  {}", resp.dump(2));
  RCLCPP_INFO(node_ptr_->get_logger(), info_msg.c_str());

  // Return the output  
  result->info = "Success";
  // result->prompt_tokens = resp["usage"]["prompt_tokens"];
  // result->completion_tokens = resp["usage"]["completion_tokens"];
  // result->total_tokens = resp["usage"]["total_tokens"];
  // for (auto& choice : resp["choices"]) {
  //   llm_if_idl::msg::ChatMessage c;
  //   c.content = choice["message"]["content"];
  //   c.role = choice["message"]["role"];
  //   result->choices.push_back(c);
  // }
  llm_if_idl::msg::ChatMessage c;
  c.content = resp["message"]["content"];
  c.role = resp["message"]["role"];
  result->choices.push_back(c);



  goal_handle->succeed(result);
  return;
}

bool CortexInterfaceNode::startModel(const std::string& model_id, std::string& response)
{
  return true;
  std::string url = fmt::format("http://localhost:1337/v1/models/{}/start", model_id); 
  json payload = {
      {"prompt_template", "<|system|>\n{system_message}\n<|user|>\n{prompt}\n<|assistant|>"},
      {"stop", json::array()},
      {"ngl", 4096},
      {"ctx_len", 4096},
      {"cpu_threads", 10},
      {"n_batch", 2048},
      {"caching_enabled", true},
      {"grp_attn_n", 1},
      {"grp_attn_w", 512},
      {"mlock", false},
      {"flash_attn", true},
      {"cache_type", "f16"},
      {"use_mmap", true},
      {"engine", "cortex.llamacpp"}
  };

  bool ret = curl_manager_.postRequest(url, payload, response);

  if (!ret) {
    RCLCPP_WARN(node_ptr_->get_logger(), "Starting Model failed.");
    return false;
  }

  // Check http code matches excpected
  long httpCode = curl_manager_.getLastCode();
  if (httpCode != 200) {
    std::string warn_str = fmt::format("Start model failed. http code: {}.  Response:  {}",
                                httpCode, response); 
    RCLCPP_WARN(node_ptr_->get_logger(), response.c_str());
    return false;
  } 

  // Success
  models_started_.push_back(model_id);
  std::string info_str = fmt::format("Start model failed. http code: {}.  Response:  {}", 
                                httpCode, response); 
  RCLCPP_INFO(node_ptr_->get_logger(), response.c_str());
  return true;
}

bool CortexInterfaceNode::chatCompletion(const std::string& model_id, const json& chat_history,
                    std::string& response,
                    const float& temperature, const bool& stream,
                    const int& max_tokens, const float& top_p
                    )
{
  // std::string url = "http://localhost:1337/v1/chat/completions"; 
  // json payload = {
  //       {"messages", chat_history},
  //       {"model", model_id},
  //       {"stream", stream},
  //       {"max_tokens", max_tokens},
  //       {"stop", {"End"}},
  //       {"frequency_penalty", 0.2},
  //       {"presence_penalty", 0.6},
  //       {"temperature", temperature},
  //       {"top_p", top_p}
  //   };

  std::string url = "http://localhost:11434/api/chat"; 
  json payload = {
        {"messages", chat_history},
        {"model", model_id},
        {"stream", stream},
        {"max_tokens", max_tokens},
        {"stop", {"End"}},
        {"frequency_penalty", 0.2},
        {"presence_penalty", 0.6},
        {"temperature", temperature},
        {"top_p", top_p}
    };

  bool ret = curl_manager_.postRequest(url, payload, response);

  if (!ret) {
    RCLCPP_WARN(node_ptr_->get_logger(), "Chat Completion failed.");
    return false;
  }

  // Check http code matches excpected
  long httpCode = curl_manager_.getLastCode();
  if (httpCode != 200) {
    std::string warn_str = fmt::format("Chat Completion failed. http code: {}.  Response:  {}", 
                                httpCode, response); 
    RCLCPP_WARN(node_ptr_->get_logger(), response.c_str());
    return false;
  } 

  // Success
  return true;
}

/* Helper Functions */
json CortexInterfaceNode::formatChatHistory(const 
                        std::vector<llm_if_idl::msg::ChatMessage>& messages) {
  json msgs = json::array();
  for (const auto& message : messages) {
    json msg;
    msg["content"] = message.content;
    msg["role"] = message.role;
    msgs.push_back(msg);
  }
  return msgs;
}

bool CortexInterfaceNode::modelLoaded(const std::string& model_id) {
  return std::find(models_started_.begin(), models_started_.end(), model_id) 
          != models_started_.end();
}
} // end of llm_if