import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from hunav_msgs.msg import Agents
from std_msgs.msg import String
import json
import math
import os
from collections import deque

class SocialWalkerInterface(Node):
    def __init__(self):
        super().__init__('social_walker_interface_v2')
        
        # 1. Trajectory Window Setup
        self.window_size = 200
        self.robot_history = deque(maxlen=self.window_size)
        self.current_robot_pose = None
        self.initialized = False
        
        # 2. File Saving Logic (For Offline Test)
        self.file_saved = False
        self.output_filename = "output_sample.json"

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/task_generator_node/jackal/odom', self.odom_callback, 10)
        self.ped_sub = self.create_subscription(Agents, '/task_generator_node/human_states', self.ped_callback, 10)
        
        # Publisher
        self.state_pub = self.create_publisher(String, '/socialwalker_state', 10)
        self.get_logger().info(f"Interface started. It will save the first 20-point frame to {self.output_filename}")

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        self.current_robot_pose = pos
        
        new_point = {
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
            "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        }

        # Ensure the window is always full (Padding)
        if not self.initialized:
            for _ in range(self.window_size):
                self.robot_history.append(new_point)
            self.initialized = True
        else:
            self.robot_history.append(new_point)

    def ped_callback(self, msg):
        if self.current_robot_pose is None:
            return

        pedestrian_list = []
        for agent in msg.agents:
            dist = math.sqrt(
                (agent.position.position.x - self.current_robot_pose.x)**2 + 
                (agent.position.position.y - self.current_robot_pose.y)**2
            )

            pedestrian_list.append({
                "id": agent.id,
                "x": round(agent.position.position.x, 3),
                "y": round(agent.position.position.y, 3),
                "vx": round(agent.velocity.linear.x, 3),
                "vy": round(agent.velocity.linear.y, 3),
                "distance": round(dist, 3)
            })

        # Final JSON Schema
        social_state = {
            "robot_past": list(self.robot_history),
            "pedestrians": pedestrian_list
        }

        # SAVE TO FILE ONCE (Offline Test requirement)
        if not self.file_saved and len(self.robot_history) == self.window_size:
            with open(self.output_filename, 'w') as f:
                json.dump(social_state, f, indent=4)
            self.get_logger().info(f"SUCCESSFULLY SAVED OFFLINE DATA TO: {os.path.abspath(self.output_filename)}")
            self.file_saved = True

        # Publish live
        output_msg = String()
        output_msg.data = json.dumps(social_state)
        self.state_pub.publish(output_msg)

def main():
    rclpy.init()
    node = SocialWalkerInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()