# import argparse
import json
from glob import glob
from typing import Any, Dict, List

# ROS
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory

import roslibpy

from llm_if_idl.action import ChatCompletion
from llm_if_idl.msg import ChatMessage
from llm_if_demos.chatbot import ChatBot


def append_service(
    client: roslibpy.Ros, name: str, services: Dict[str, roslibpy.Service]
) -> Dict[str, roslibpy.Service]:
    """Update current services with the required one.

    Args:
        client (roslibpy.Ros): ROS client.
        name (str): Required service name.
        services (Dict[str, roslibpy.Service]): Dictionary of current services.

    Returns:
        Dicr[str, roslibpy.Service]: Updated dictionary of services.
    """
    if name not in services:
        services[name] = roslibpy.Service(client, name, client.get_service_type(name))
    return services


class TurtlebotLLMController(Node):

    def _get_sys_prompt(self):
        api_dir = get_package_share_directory('llm_if_demos')

        self.api = []
        for api_file in glob(f"{api_dir}/*.json"):
            with open(api_file, "r") as f:
                self.api.append(json.load(f))

        sys_msg = ChatMessage(role="system", content="")
        sys_msg.content += r"""You are managing controls for turtles in a ROS2 simulator.\n"""
        sys_msg.content += r"""Your objective is to understand user requests and transform them into """ \
                                         r"""api calls used by the simulator.\n"""
        sys_msg.content += r"""Use this JSON schema to achieve the user's goals:\n\n"""
        sys_msg.content += str(self.api) + "\n\n"
        sys_msg.content += r"""Replace the '\$TURTLE_NAME' in the service field with """ \
                                 r"""the name of the turlte you want to control.\n"""
        sys_msg.content += r"""Do not include explanations or conversation in the response."""

        return [sys_msg]

    def __init__(self, node_name='turtlebot_llm_controller', host="localhost", port=9090):
        """
        Initialize the node and connection to roslibpy.
        
        Parameters:
            node_name (str): Name of the ROS2 node. Default is 'obj_det'.
            host (str): ROS host.
            node_name (str): ROS port.
        """
        super().__init__(node_name=node_name)

        # create a ROS client
        self.ros_client = roslibpy.Ros(host=host, port=port)
        self.ros_client.run()

        # interface with the chat completion server
        self.sys_prompt = self._get_sys_prompt()
        self.chatbot = ChatBot(node_name=node_name, sys_instruction=self.sys_prompt)

    def run(self):
        self.get_logger().info("Running cmd line prompt service...")
        while rclpy.ok():
            prompt = input("Enter a prompt: ")

            self.get_logger().info("Generating API calls. This may take some time...")
            succ, err_msg, resp = self.chatbot(prompt)
            if not succ:
                self.get_logger.warn(f"Chatbot Error:  {err_msg}")
                continue

            self.get_logger().info(f'Chatbot Response:\n{resp}')

            generated_api_calls = self.remove_lines_starting_with_ticks(resp)
            generated_api_calls = self.post_process_response_(generated_api_calls)
            services = {}
            for call in generated_api_calls:
                # get required service (in case they changed)
                self.get_logger().info("Getting required service. This might take some time...")
                services = append_service(self.ros_client, call["service"], services)
                self.get_logger().info("Done.")

                try:
                    self.get_logger().info(
                        "Calling service {} with args {}".format(
                            call["service"], call["args"]
                        )
                    )
                    input("Press Enter to continue...")
                    service = services[call["service"]]
                    request = roslibpy.ServiceRequest(call["args"])
                    service.call(request)
                except Exception as e:
                    self.get_logger().warn(f"Failed to call service with {e}.")

    def remove_lines_starting_with_ticks(self, response: str) -> str:
        """
        phi3 is sometimes outputing data in format like:
            ```json
            [
              {"service": "/spawn", "args": {"x": 10, "y": 10, "name": "turtle2"}},
            ]
            ```
        Remove any lines starting with ``` (ticks).

        Args:
            response (str): Chatbot response.

        Returns:
            response (str): Chatbot response after removing ticks.
        """
        # Split the string into lines
        lines = response.splitlines()
        
        # Filter out lines that start with '''
        filtered_lines = [line for line in lines if not line.startswith("```")]
        
        # Join the filtered lines back into a single string
        response = "\n".join(filtered_lines)
        return response

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
