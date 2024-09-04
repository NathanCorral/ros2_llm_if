import requests
import json

"""
run:
$ cortex
"""


# Function to send a chat message
def chat(model_id: str, msg: str):
    url = "http://localhost:1337/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "<|user|>",
                "content": msg
            },
        ],
        "stream": False,
        "max_tokens": 50,
        "stop": [
            "End"
        ],
        "frequency_penalty": 0.2,
        "presence_penalty": 0.6,
        "temperature": 0.8,
        "top_p": 0.95
    }
    
    response = requests.post(url, headers=headers, json=payload)
    # print("Response Status Code:", response.status_code)

    if response.status_code != 200:
        response.raise_for_status()
    # print(response.text)
    return response.json()


# Function to send a chat message
def chat_stream(model_id: str, msg: str):
    url = "http://localhost:1337/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "<|user|>",
                "content": msg
            },
        ],
        "stream": True,
        "max_tokens": 50,
        "stop": [
            "End"
        ],
        "frequency_penalty": 0.2,
        "presence_penalty": 0.6,
        "temperature": 0.8,
        "top_p": 0.95
    }
    
    response = requests.post(url, headers=headers, json=payload)
    # print("Response Status Code:", response.status_code)

    if response.status_code != 200:
        response.raise_for_status()
    try:
        response_data = response.json()
        print("Response Body:", response_data)
    except requests.exceptions.JSONDecodeError:
        # print("The response is not in JSON format or is empty.")
        pass

    # Try parsing plain text hack
    response_data = response.text
    # print("Response data:  ",  response_data)
    try:
        response_data_temp = response.text.split("data: ")
        # response_data_temp = [json.loads(word) for word in response_data_temp]
        response_data_list = []
        for word in response_data_temp:
            # print("word:  ",  word)
            word = word.strip()
            if len(word) == 0:
                continue
            if word == "[DONE]":
                break
            next_chunk = json.loads(word)
            if next_chunk["choices"][0]["delta"]["content"] == "<|end|>":
                break
            # print("Content:  ",  next_chunk["choices"][0]["delta"]["content"])

            response_data_list.append(next_chunk)

        response_data = response_data_list
    except json.decoder.JSONDecodeError:
        print("Hack decode error")
        pass

    # print("DONE")
    return response_data
    

def pull_model(model_id: str):
    url = f"http://localhost:1337/v1/models/{model_id}/pull"
    response = requests.post(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def start_model(model_id: str):
    url = f"http://localhost:1337/v1/models/{model_id}/start"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt_template": "<|system|>\n{system_message}\n<|user|>\n{prompt}\n<|assistant|>",
        "stop": [],
        "ngl": 4096,
        "ctx_len": 4096,
        "cpu_threads": 10,
        "n_batch": 5,
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

def get_single_word_resp(config, msg, stream=False):
    model_id = config["model_id"]

    if stream:
        chat_response = chat_stream(model_id, msg)
        chat_content = []
        for item in chat_response:
            choices = item["choices"]
            best = choices[0]["delta"]["content"].strip()
            if len(best) == 0:
                continue
            if best == "<|end|>":
                break
            chat_content.append(best)
        single_word = "".join(chat_content)
    else:
        chat_response = chat(model_id, msg)
        # print(chat_response)
        single_word = chat_response["choices"][0]["message"]["content"].split("<|end|>")[0]

    return single_word


if __name__ == "__main__":
    config = {"model_id": "phi3"}
    model_id = config["model_id"]

    # model_id = "phi3:mini-gguf"
    names = ["Sam", "Bart", "Chatgtp", "Dough", "Terminator", "Bosco", "Clifford", "Nate"]
    msg0 = f'Pick a nickname from the list: [{", ".join(names)}]. Restate only your name and nothing else.'
    msg1 = f'Pick two nicknames from the list: [{", ".join(names)}]. Restate only the names, separated by a space, and nothing else.'
    msg2 = f'Pick a first and last name from the list: [{", ".join(names)}]. Restate only the names and nothing else.'
    msgs = [msg0]*5

    # Starting the model
    pull_model(model_id)
    # print(f"Model '{model_id}' pulled")
    model_response = start_model(model_id)
    # print("Model Start Response:", model_response)

    responses = []
    for msg in msgs:
        response = get_single_word_resp(config, msg)
        responses.append(response)#

    print(responses)

