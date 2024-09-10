import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from llm_if_idl.action import ChatCompletion
from llm_if_idl.msg import ChatMessage
from llm_if_idl.action._chat_completion import ChatCompletion_FeedbackMessage


from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future

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
}

class ChatBot(Node):
    sys_instruction = personalities["robot"]

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name=node_name)

        # llm_if_server Chat Completion client
        self.cc_client = ActionClient(self, ChatCompletion, '/cortex_chat_completion')

        while not self.cc_client.wait_for_server(1):
            self.get_logger().warn(
                f"Waiting for {self.cc_client._action_name} action server.."
            )
        self.get_logger().info(
            f"Action server {self.cc_client._action_name} found."
        )

        self.get_logger().info(self.info_string())

        self.chat_history = [self.sys_instruction]

        # Chat default settings
        self.model = "phi3:mini-gguf"
        self.temperature = 0.8
        self.max_tokens = 2048
        self.top_p = 0.95

        self.chat_completed_flag = True

    def chat(self) -> bool:
        """
        :return: Flag to continue the program (False == exit)
        """
        prompt = input("> ")

        # Handle special prompts
        if (prompt.lower() == "q"):
            return False
        elif (prompt.lower() == "_clear"):
            self.chat_history = [self.sys_instruction]
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

    def on_feedback(self, feedback_msg: ChatCompletion_FeedbackMessage) -> None:
        print(f"{feedback_msg.feedback.status}", end="")

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

        if result.info.lower() != "success":
            self.get_logger().warn(f'Failed interface with error:  {result.info}')
            return

        self.chat_history += [result.choices[0]]
        self.get_logger().info(f'{self.chat_history[-1].content}')


    def info_string(self) -> str:
        return (
            "\n\n"
            "\tStarting Chatbot demo.\n"
            "\tEnter '_clear' to clear chat history.\n"
            "\tEnter 'q' to exit.\n"
        )

def main(args=None):
    rclpy.init(args=args)
    chatbot = ChatBot(node_name="chatbot")
    while rclpy.ok():
        if not chatbot.chat():
            break

        while rclpy.ok() and not chatbot.chat_completed_flag:
            rclpy.spin_once(chatbot, timeout_sec=0.1)
    chatbot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
