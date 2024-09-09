#ifndef CURLMANAGER_H
#define CURLMANAGER_H

#include <string>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace llm_if {
class CurlManager {
public:
  CurlManager();
  ~CurlManager();

  // Sends a POST request to the specified URL with JSON data.
  bool postRequest(const std::string& url, const nlohmann::json& payload, std::string& response);

  // Get the last http Code
  long getLastCode();

private:
  // Stores the CURL handle
  // std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl_handle_prt_;
  CURL* curl_handle_;

  // Static helper function for libcurl to write the response data
  static size_t writeCallback(void* contents, size_t size, size_t nmemb, std::string* response);
};
} // end of namespace llm_if

#endif // CURLMANAGER_H
