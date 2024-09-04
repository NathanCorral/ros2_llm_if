#include "llm_if_server/cortex_chat_completion_node.hpp"

namespace llm_if {
CortexInterfaceNode::CortexInterfaceNode(const rclcpp::Node::SharedPtr node_ptr)
    : node_ptr_(node_ptr) {
  declare_parameters_();

  // auto cb_group = node_ptr_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  // rclcpp::SubscriptionOptions options;
  // options.callback_group = cb_group;

  // // audio subscription
  // audio_sub_ = node_ptr_->create_subscription<std_msgs::msg::Int16MultiArray>(
  //     "audio", 5, std::bind(&InferenceNode::on_audio_, this, std::placeholders::_1), options);

  // inference action server
  inference_action_server_ = rclcpp_action::create_server<ChatCompletion>(
      node_ptr_, "cortex_chat_completion",
      std::bind(&CortexInterfaceNode::on_inference_, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&CortexInterfaceNode::on_cancel_inference_, this, std::placeholders::_1),
      std::bind(&CortexInterfaceNode::on_inference_accepted_, this, std::placeholders::_1));

  // // parameter callback handle
  // on_parameter_set_handle_ = node_ptr_->add_on_set_parameters_callback(
  //     std::bind(&InferenceNode::on_parameter_set_, this, std::placeholders::_1));

  // // whisper
  // model_manager_ = std::make_unique<ModelManager>();
  // batched_buffer_ = std::make_unique<BatchedBuffer>(
  //     std::chrono::seconds(node_ptr_->get_parameter("batch_capacity").as_int()),
  //     std::chrono::seconds(node_ptr_->get_parameter("buffer_capacity").as_int()),
  //     std::chrono::milliseconds(node_ptr_->get_parameter("carry_over_capacity").as_int()));
  // whisper_ = std::make_unique<Whisper>();

  RCLCPP_INFO(node_ptr_->get_logger(), "Cortex Server Initialized.");


  // initialize_whisper_();
}

void CortexInterfaceNode::declare_parameters_() {
  // buffer parameters
  node_ptr_->declare_parameter("model_id", "phi3:mini-gguf");
  // node_ptr_->declare_parameter("model_id", "phi3:mini-gguf");
  // node_ptr_->declare_parameter("model_id", "phi3:mini-gguf");

  // node_ptr_->declare_parameter("batch_capacity", 6);
  // node_ptr_->declare_parameter("buffer_capacity", 2);
  // node_ptr_->declare_parameter("carry_over_capacity", 200);

  // // whisper parameters
  // node_ptr_->declare_parameter("model_name", "base.en");
  // // consider other parameters:
  // // https://github.com/ggerganov/whisper.cpp/blob/a4bb2df36aeb4e6cfb0c1ca9fbcf749ef39cc852/whisper.h#L351
  // node_ptr_->declare_parameter("wparams.language", "en");
  // node_ptr_->declare_parameter("wparams.n_threads", 4);
  // node_ptr_->declare_parameter("wparams.print_progress", false);
  // node_ptr_->declare_parameter("cparams.flash_attn", true);
  // node_ptr_->declare_parameter("cparams.gpu_device", 0);
  // node_ptr_->declare_parameter("cparams.use_gpu", true);
}

// void CortexInterfaceNode::initialize_whisper_() {
  // std::string model_name = node_ptr_->get_parameter("model_name").as_string();
  // RCLCPP_INFO(node_ptr_->get_logger(), "Checking whether model %s is available...",
  //             model_name.c_str());
  // if (!model_manager_->is_available(model_name)) {
  //   RCLCPP_INFO(node_ptr_->get_logger(), "Model %s is not available. Attempting download...",
  //               model_name.c_str());
  //   if (model_manager_->make_available(model_name) != 0) {
  //     std::string err_msg = "Failed to download model " + model_name + ".";
  //     RCLCPP_ERROR(node_ptr_->get_logger(), err_msg.c_str());
  //     throw std::runtime_error(err_msg);
  //   }
  //   RCLCPP_INFO(node_ptr_->get_logger(), "Model %s downloaded.", model_name.c_str());
  // }
  // RCLCPP_INFO(node_ptr_->get_logger(), "Model %s is available.", model_name.c_str());

  // language_ = node_ptr_->get_parameter("wparams.language").as_string();
  // whisper_->wparams.language = language_.c_str();
  // whisper_->wparams.n_threads = node_ptr_->get_parameter("wparams.n_threads").as_int();
  // whisper_->wparams.print_progress = node_ptr_->get_parameter("wparams.print_progress").as_bool();
  // whisper_->cparams.flash_attn = node_ptr_->get_parameter("cparams.flash_attn").as_bool();
  // whisper_->cparams.gpu_device = node_ptr_->get_parameter("cparams.gpu_device").as_int();
  // whisper_->cparams.use_gpu = node_ptr_->get_parameter("cparams.use_gpu").as_bool();

  // RCLCPP_INFO(node_ptr_->get_logger(), "Initializing model %s...", model_name.c_str());
  // whisper_->initialize(model_manager_->get_model_path(model_name));
  // RCLCPP_INFO(node_ptr_->get_logger(), "Model %s initialized.", model_name.c_str());
// }

// rcl_interfaces::msg::SetParametersResult
// CortexInterfaceNode::on_parameter_set_(const std::vector<rclcpp::Parameter> &parameters) {
//   rcl_interfaces::msg::SetParametersResult result;
//   for (const auto &parameter : parameters) {
//     if (parameter.get_name() == "n_threads") {
//       whisper_->wparams.n_threads = parameter.as_int();
//       RCLCPP_INFO(node_ptr_->get_logger(), "Parameter %s set to %d.", parameter.get_name().c_str(),
//                   whisper_->wparams.n_threads);
//       continue;
//     }
//     result.reason = "Parameter " + parameter.get_name() + " not handled.";
//     result.successful = false;
//     RCLCPP_WARN(node_ptr_->get_logger(), result.reason.c_str());
//   }
//   result.successful = true;
//   return result;
// }

// void InferenceNode::on_audio_(const std_msgs::msg::Int16MultiArray::SharedPtr msg) {
//   batched_buffer_->enqueue(msg->data);
// }

rclcpp_action::GoalResponse
CortexInterfaceNode::on_inference_(const rclcpp_action::GoalUUID & /*uuid*/,
                             std::shared_ptr<const ChatCompletion::Goal> /*goal*/) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Received inference request.");
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
CortexInterfaceNode::on_cancel_inference_(const std::shared_ptr<GoalHandleChatCompletion> /*goal_handle*/) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Cancelling inference...");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void CortexInterfaceNode::on_inference_accepted_(const std::shared_ptr<GoalHandleChatCompletion> goal_handle) {
  RCLCPP_INFO(node_ptr_->get_logger(), "Starting inference...");
  auto feedback = std::make_shared<ChatCompletion::Feedback>();
  auto result = std::make_shared<ChatCompletion::Result>();
  chat_completion_start_time_ = node_ptr_->now();


  while (rclcpp::ok()) {
    if (node_ptr_->now() - chat_completion_start_time_ > goal_handle->get_goal()->max_duration) {
      result->info = "Inference timed out.";
      RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->succeed(result);
      // batched_buffer_->clear();
      return;
    }

    if (goal_handle->is_canceling()) {
      result->info = "Inference cancelled.";
      RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
      goal_handle->canceled(result);
      // batched_buffer_->clear();
      return;
    }

    std::string model_id = node_ptr_->get_parameter("model_id").as_string();
    feedback->status = "Running query on model:  " + model_id;
    result->completion_tokens = goal_handle->get_goal()->max_tokens;


  //   // run inference
  //   auto transcription = inference_(batched_buffer_->dequeue());

  //   // feedback to client
  //   feedback->transcription = transcription;
  //   feedback->batch_idx = batched_buffer_->batch_idx();
  //   goal_handle->publish_feedback(feedback);

  //   // update inference result
  //   // if (result->transcriptions.size() ==
  //   //     static_cast<std::size_t>(batched_buffer_->batch_idx() + 1)) {
  //   //   result->transcriptions[result->transcriptions.size() - 1] = feedback->transcription;
  //   // } else {
  //   //   result->transcriptions.push_back(feedback->transcription);
  //   // }
  }

  if (rclcpp::ok()) {
    result->info = "Inference succeeded.";
    RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
    goal_handle->succeed(result);
    // batched_buffer_->clear();
  }
}

// std::string InferenceNode::inference_(const std::vector<float> &audio) {
//   auto inference_start_time = node_ptr_->now();
//   auto transcription = whisper_->forward(audio);
//   auto inference_duration =
//       (node_ptr_->now() - inference_start_time).to_chrono<std::chrono::milliseconds>();
//   if (inference_duration > whisper::count_to_time(audio.size())) {
//     RCLCPP_WARN(node_ptr_->get_logger(),
//                 "Inference took longer than audio buffer size. This leads to un-inferenced audio "
//                 "data. Consider increasing thread number or compile with accelerator support.");
//   }
//   return transcription;
// }
} // end of llm_if