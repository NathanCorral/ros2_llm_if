import json
import requests

from typing import Any, Dict, List

class CortexClient:
    def __init__(self, model_id: str = "phi3:mini-gguf", 
                    ip: str = "localhost", 
                    port: str = "1337"):
        self.model_id = model_id
        self.ip, self.port = ip, port
        self.url = f'http://{ip}:{port}/v1'

        self.pull_model()
        self.start_model()

    def start_model(self):
        url = f'{self.url}/models/{self.model_id}/start'
        headers = {
        "Content-Type": "application/json"
        }
        # Default Payload from:
        #   https://cortex.so/api-reference/#tag/models/post/v1/models/{modelId}/start
        # modified prompt_template for phi3:
        #   https://arxiv.org/abs/2404.14219
        payload = {
            "prompt_template": "<|system|>\n{system_message}\n<|user|>\n{prompt}\n<|assistant|>",
            "stop": [],
            "ngl": 4096,
            "ctx_len": 4096,
            "cpu_threads": 10,
            "n_batch": 2048,
            "caching_enabled": True,
            "grp_attn_n": 1,
            "grp_attn_w": 512,
            "mlock": False,
            "flash_attn": True,
            "cache_type": "f16",
            "use_mmap": True,
            "engine": "cortex.llamacpp"
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def pull_model(self):
        url = f'{self.url}/models/{self.model_id}/pull'
        response = requests.post(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def chat_completion(self, msgs, temperature, max_tokens : int = 4096):
        """
        Call: 
            https://cortex.so/api-reference/#tag/inference
        """
        url = f'{self.url}/chat/completions'
        headers = {
            "Content-Type": "application/json"
        }

        if not isinstance(msgs, list):
            msgs = [msgs]

        payload = {
            "model": self.model_id,
            "messages": msgs,
            "stream": False,
            "max_tokens": max_tokens,
            "stop": [
                "End"
            ],
            "frequency_penalty": 0.2,
            "presence_penalty": 0.6,
            "temperature": temperature,
            "top_p": 0.95
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            response.raise_for_status()

        return response.json()



class CortexInterface:
    def __init__(self, api: Any, key: str = None, model_id: str = "phi3:mini-gguf") -> None:
        self.api_ = api
        self.system_prompt_ = f"\
            Use this JSON schema to achieve the user's goals:\n\
            {str(api)}\n\
            Respond as a list of JSON objects.\
            Replace the 'turtle_name' in the service field with the actual name of the turtle you want to apply the action to.\
            Do not include explanations or conversation in the response.\
        "
        self.chat_history_ = []
        self.client = CortexClient(model_id=model_id)

    def prompt_to_api_calls(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.8,
    ) -> List[Dict]:
        """Turns prompt into API calls.

        Args:
            prompt (str): Prompt.
            model (str, optional): Specified in construction
            temperature (float, optional): Generation temperature. Defaults to 0.8.

        Returns:
            Dict: API calls.
        """
        self.chat_history_.append(  # prompt taken from https://github.com/Significant-Gravitas/Auto-GPT/blob/master/autogpt/prompts/default_prompts.py
            {
                "role": "<|user|>",
                "content": f"\
                    {prompt}\n\
                    Respond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\
                ",
            }
        )
        msgs = [{"role": "<|system|>", "content": self.system_prompt_}] + self.chat_history_
        # for msg in msgs:
        #     print("Msg:")
        #     print(msg)
        #     print()
        # return []

        try:
            response = self.client.chat_completion(
                msgs=msgs,
                temperature=temperature,
            )
        except Exception as e:
            print(f"Oops! Something went wrong with {e}.")
            self.chat_history_.pop()
            return []

        self.chat_history_.append(
            {"role": "<|assistant|>", "content": response["choices"][0]["message"]["content"].split("<|end|>")[0]}
        )

        content = self.chat_history_[-1]["content"]
        print(f"Got response:\n{content}")
        return self.post_process_response_(content)

    def remove_lines_starting_with_ticks(self, gpt_response: str) -> str:
        """
        phi3 is sometimes outputing data in format like:
            ```json
            [
              {"service": "/spawn", "args": {"x": 10, "y": 10, "name": "turtle2"}},
            ]
            ```
        Remove any lines starting with ``` (ticks).

        Args:
            gpt_response (str): GPT response.

        Returns:
            gpt_response (str): GPT response after removing ticks.
        """
        # Split the string into lines
        lines = gpt_response.splitlines()
        
        # Filter out lines that start with '''
        filtered_lines = [line for line in lines if not line.startswith("```")]
        
        # Join the filtered lines back into a single string
        gpt_response = "\n".join(filtered_lines)
        return gpt_response

    def post_process_response_(self, gpt_response: str) -> List[Dict]:
        """Applies some simple post-processing to the model response.

        Args:
            gpt_response (str): GPT response.

        Returns:
            List[Dict]: Post-processed response.
        """
        gpt_response = self.remove_lines_starting_with_ticks(gpt_response)
        gpt_response = gpt_response.replace("'", '"')
        gpt_response = json.loads(gpt_response)

        if isinstance(gpt_response, list):
            return gpt_response
        else:
            return [gpt_response]
