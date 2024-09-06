#include "llm_if_server/cortex_chat_completion_node.hpp"

namespace llm_if {
CortexInterfaceNode::CortexInterfaceNode(const rclcpp::Node::SharedPtr node_ptr)
    : node_ptr_(node_ptr) {
  declare_parameters_();
  if(try_connect_()) {
      RCLCPP_INFO(node_ptr_->get_logger(), "Model Started.");
  }

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


  


  RCLCPP_INFO(node_ptr_->get_logger(), "Cortex Server Initialized :=).");


  // initialize_whisper_();
}

bool CortexInterfaceNode::try_connect_() {
  CURL* curl;
  CURLcode res;
  std::string url;
  std::string jsonData;
  std::string responseString;
  bool ret = false;
  struct curl_slist* headers = NULL;
  std::string model_id = node_ptr_->get_parameter("model_id").as_string();

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  if(curl) {
    // Define URL and headers
    url = fmt::format("http://localhost:1337/v1/models/{}/start", model_id); 
    headers = curl_slist_append(headers, "Content-Type: application/json");

    // Create JSON payload
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

    jsonData = payload.dump(); // Serialize payload to string

    // Set CURL options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.c_str());

    // Response handling
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseString);

    // Perform the request
    res = curl_easy_perform(curl);

    if(res != CURLE_OK) {
      std::string error_reason = fmt::format("Starting Model Failed, will retry: {}", curl_easy_strerror(res)); 
      RCLCPP_WARN(node_ptr_->get_logger(), error_reason.c_str());
    } else {
      // HTTP status code
      long httpCode(0);
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

      if (httpCode == 200) {
        std::string response = fmt::format("Response Data: {}", responseString); 
        RCLCPP_INFO(node_ptr_->get_logger(), response.c_str());
        ret = true;
      } else {
        std::string response = fmt::format("Request failed with status code: {}", httpCode); 
        RCLCPP_WARN(node_ptr_->get_logger(), response.c_str());
      }
    }

    // Cleanup
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
  }

  curl_global_cleanup();
  return ret;
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
    // if (node_ptr_->now() - chat_completion_start_time_ > goal_handle->get_goal()->max_duration) {
    //   result->info = "Inference timed out.";
    //   RCLCPP_INFO(node_ptr_->get_logger(), result->info.c_str());
    //   goal_handle->succeed(result);
    //   // batched_buffer_->clear();
    //   return;
    // }

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