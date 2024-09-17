import json
from glob import glob

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from llm_if_idl.action import ChatCompletion
from llm_if_idl.msg import ChatMessage
from llm_if_idl.action._chat_completion import ChatCompletion_FeedbackMessage


def get_personalities():
    api_dir = get_package_share_directory('llm_if_demos')
    api = []
    for api_file in glob(f"{api_dir}/*.json"):
        with open(api_file, "r") as f:
            api.append(json.load(f))

    personalities = {
        "robot": ChatMessage(role="system", \
                        content="You are a professional assistant.\n\
                        Reply only with the required information.\n\
                        Do not interact with the user.\n\
                        Do not explain things.\n\
                        If there is something you do not know/understand/relate to as an AI,\
                                simply respond with 'I dont know' or 'I can't assist with that'.\n\
                        "),
        "base": ChatMessage(role="system", \
                        content="You are a helpful assistant.\n\
                        "),
        "actor": ChatMessage(role="system", \
                        content=f"\
            Use this JSON schema to achieve the user's goals:\n\
            {str(api)}\n\
            Respond as a list of JSON objects.\n\
            Replace the 'turtle_name' in the service field with the actual \
            name of the turtle you want to control.\n\
            Do not include explanations or conversation in the response.\n\
            ",
        ),
    }
    return personalities

class ChatBot(Node):
    def __init__(self, node_name: str, sys_instruction: str) -> None:
        super().__init__(node_name=node_name)
        if isinstance(sys_instruction, list):
            self.sys_instruction = sys_instruction
        else:
            self.sys_instruction = [sys_instruction]
        self.clear()

        # llm_if_server Chat Completion client
        group = MutuallyExclusiveCallbackGroup()
        self.cc_client = ActionClient(self, ChatCompletion, '/cortex_chat_completion', callback_group=group)

        while not self.cc_client.wait_for_server(1):
            self.get_logger().warn(
                f"Waiting for {self.cc_client._action_name} action server.."
            )
        self.get_logger().info(
            f"Action server {self.cc_client._action_name} found."
        )

        # Chat default settings
        self.model = "phi3:mini-gguf"
        self.model = "phi3"
        self.temperature = 0.8
        self.max_tokens = 2048
        self.top_p = 0.95

        self.chat_completed_flag = True
        self.last_result = None

    def clear(self):
        """
        Clear the chat history
        """
        self.chat_history = self.sys_instruction

    def __call__(self, prompt):
        """
        Interface with the action server.
        Return when the result is complete
        :parameter prompt:  user prompt for the LLM
        :return: (success: bool, error_msg: str, response: str)
        """
        success = False
        resp = None
        error_msg = ""
        if not self.chat_completed_flag:
            return success, "Current chat in progress", resp
        self.chat_completed_flag = False

        self.start_cc(prompt)
        """
        I would like to use
            rclpy.spin_until_future_complete(self, future)
        but it blocks even after call is completed.
        """
        while rclpy.ok() and not self.chat_completed_flag:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.last_result.info.lower() != "success":
            error_msg = last_result.info
        else:
            resp = self.chat_history[-1].content
            # Remove <|end|>
            resp = resp.split("<|end|>")[0]
            success = True

        return success, error_msg, resp            

    def chat(self) -> bool:
        """
        :return: Flag to continue the program (False == exit)
        """
        try:
            prompt = input("> ")
        except EOFError:
            self.get_logger().warn(f'--Caught [EOF], shutting down..')
            return False

        # Handle special prompts
        if (prompt.lower() == "q"):
            return False
        elif (prompt.lower() == "_clear"):
            self.clear()
            return self.chat() # re-run prompt

        self.chat_completed_flag = False
        self.start_cc(prompt)
        return True

    def start_cc(self, prompt):
        self.chat_history += [ChatMessage(role="user", content=prompt)]

        goal = ChatCompletion.Goal()
        goal.messages = self.chat_history
        goal.model = self.model
        goal.temperature = self.temperature
        goal.max_tokens = self.max_tokens
        goal.top_p = self.top_p

        future = self.cc_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        future.add_done_callback(self.on_goal_accepted)
        return future

    def on_feedback(self, feedback_msg: ChatCompletion_FeedbackMessage) -> None:
        """
        todo
        """
        self.get_logger().info(f"{feedback_msg.feedback.status}")

    def on_goal_accepted(self, future: Future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.chat_completed_flag = True
            return
        future = goal_handle.get_result_async()
        future.add_done_callback(self.on_done)

    def on_done(self, future: Future) -> None:
        result: ChatCompletion.Result = future.result().result
        self.chat_completed_flag = True
        self.last_result = result

        if result.info.lower() != "success":
            self.get_logger().warn(f'Failed interface with error:  {result.info}')
            return

        self.chat_history += [result.choices[0]]
        # self.get_logger().info(f'{self.chat_history[-1].content}')

    def print_last_result(self):
        if self.last_result.info.lower() != "success":
            return
        self.get_logger().info(f'{self.chat_history[-1].content}')

    def info_string(self) -> str:
        inst = [f"{x.content}\n" for x in self.sys_instruction]
        return (
            "Chatbot created with System Instructions:\n\n"
            f"{inst}\n"
            "\n\n"
            "\tStarting Chatbot demo.\n"
            "\tEnter '_clear' to clear chat history.\n"
            "\tEnter 'q' to exit.\n"
        )

def main(args=None):
    rclpy.init(args=args)

    personalities = get_personalities()
    sys_instruction = personalities["robot"]
    chatbot = ChatBot(node_name="chatbot", sys_instruction=sys_instruction)
    chatbot.get_logger().info(chatbot.info_string())
    while rclpy.ok():
        if not chatbot.chat():
            break

        while rclpy.ok() and not chatbot.chat_completed_flag:
            rclpy.spin_once(chatbot, timeout_sec=0.1)
        
        chatbot.print_last_result()

    chatbot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
