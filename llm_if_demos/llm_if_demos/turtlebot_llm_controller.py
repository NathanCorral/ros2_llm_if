# import argparse
# import json
# from glob import glob

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import roslibpy

from llm_if_idl.action import ChatCompletion
from llm_if_idl.msg import ChatMessage


class TurtlebotLLMController(Node):
    def __init__(self, node_name='turtlebot_llm_controller', host="localhost", port=9090):
        """
        Initialize the node and connection to roslibpy.
            # LLM parameters todo
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'obj_det'.
            host (str): ROS host.
            node_name (str): ROS port.
        """
        super().__init__(node_name=node_name)

        # create a ROS client
        # self.ros_client = roslibpy.Ros(host=host, port=port)
        # self.ros_client.run()
        self.log = self.get_logger()

        # interface with the chat completion server
        chat_completion_callback_group = MutuallyExclusiveCallbackGroup()
        chat_completion_callback_group = None
        self.chat_completion_client = ActionClient(self, ChatCompletion, '/llm_if/cortex_chat_completion', callback_group=chat_completion_callback_group)
        req_services = [self.chat_completion_client]
        while not all([r.wait_for_server(timeout_sec=1.0) for r in req_services]):
            self.log.info('Service not available, waiting...')


    def run(self):
        self.log.info("Running cmd line prompt service...")
        while rclpy.ok():
            prompt = input("Enter a prompt: ")
            print("you said:   ",  prompt)

            goal = ChatCompletion.Goal()
            msg0 = ChatMessage(role="system", content="You are a helful assistant.\nYour name is Pokedex.\nKeep your response breif.")
            msg1 = ChatMessage(role="user", content="Introduce Yourself")
            goal.messages = [msg0, msg1]
            goal.model = "phi3:mini-gguf"
            goal.model = "phi3:mini-gguf"
            goal.tempurature = 0.7
            future = self.chat_completion_client.send_goal_async(goal)
            print("I sent the shit")
            rclpy.spin_until_future_complete(self, future)
            print("Got response i cant believe it!")




def main(args=None):
    rclpy.init(args=args)
    node = TurtlebotLLMController()
    
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
