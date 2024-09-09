#include <iostream>
#include <stdexcept>
#include <sstream>

#include "llm_if_server/curl_manager.hpp"

namespace llm_if {

CurlManager::CurlManager() {
  if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
    throw std::runtime_error("Failed to initialize libcurl globally.");
  }
  std::cout << "Curl Server Initialized" << std::endl;
  curl_handle_ = curl_easy_init();

  // Set up CURL options that are constant for every request
  // curl_easy_setopt(curl_handle_, CURLOPT_FOLLOWLOCATION, 1L);  // Follow redirects
}

CurlManager::~CurlManager() {
  if (curl_handle_) {
    curl_easy_cleanup(curl_handle_);
  }
  curl_global_cleanup();
  std::cout << "Curl Server Destroyed" << std::endl;
}

bool CurlManager::postRequest(const std::string& url, const nlohmann::json& jsonData, 
                                std::string& response) {
  if (!curl_handle_) {
    // Try and restart
    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
      return false;
    }
  }
  CURLcode res;

  // Convert JSON data to string
  std::string jsonStr = jsonData.dump();

  // Set CURL options for POST request
  curl_easy_setopt(curl_handle_, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl_handle_, CURLOPT_POST, 1L);
  curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDS, jsonStr.c_str());
  curl_easy_setopt(curl_handle_, CURLOPT_POSTFIELDSIZE, jsonStr.size());

  // Set headers for JSON content type
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(curl_handle_, CURLOPT_HTTPHEADER, headers);

  // Set callback function to capture response
  curl_easy_setopt(curl_handle_, CURLOPT_WRITEFUNCTION, writeCallback);
  curl_easy_setopt(curl_handle_, CURLOPT_WRITEDATA, &response);

  // Perform the request
  res = curl_easy_perform(curl_handle_);

  // Cleanup headers
  curl_slist_free_all(headers);

  // Check for errors
  if (res != CURLE_OK) {
    std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    return false;
  }

  return true;
}

long CurlManager::getLastCode() {
  long ret(0);
  if (curl_handle_) {
    curl_easy_getinfo(curl_handle_, CURLINFO_RESPONSE_CODE, &ret);
  }
  return ret;
}

size_t CurlManager::writeCallback(void* contents, size_t size, size_t nmemb, 
                  std::string* response) {
  size_t totalSize = size * nmemb;
  response->append((char*)contents, totalSize);
  return totalSize;
}

} // end of namespace llm_if
