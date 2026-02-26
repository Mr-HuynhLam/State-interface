#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import json
import math
from collections import deque

from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String

# Import the local plan evaluation message from Nav2/DWB
from dwb_msgs.msg import LocalPlanEvaluation
from rclpy.qos import qos_profile_sensor_data

class VisionStateInterface(Node):
    def __init__(self):
        super().__init__('vision_state_interface')
        
        self.Th = 8               
        self.sampling_rate = 0.1  
        
        self.robot_history_world = deque(maxlen=self.Th)
        self.ped_seq_buffer = deque(maxlen=self.Th)
        self.frame_counter = 0
        
        self.current_odom = None
        self.latest_bboxes = None
        self.latest_eval = None

        # 1. Odometry
        self.create_subscription(Odometry, '/task_generator_node/jackal/odom', self.odom_cb, qos_profile_sensor_data)
        
        # 2. Camera Bounding Boxes + Depth
        self.create_subscription(Detection2DArray, '/task_generator_node/jackal/gt_human_bboxes_2d', self.bbox_cb, qos_profile_sensor_data)
        
        # 3. DWB Local Planner Evaluation (The Candidates!)
        self.create_subscription(LocalPlanEvaluation, '/task_generator_node/jackal/evaluation', self.eval_cb, qos_profile_sensor_data)
        
        # Publisher
        self.state_pub = self.create_publisher(String, '/socialwalker_state', 10)

        self.timer = self.create_timer(self.sampling_rate, self.sync_and_publish)
        self.get_logger().info("✅ Vision State Interface Started (Now including Trajectories)!")

    def odom_cb(self, msg):
        self.current_odom = msg.pose.pose

    def bbox_cb(self, msg):
        self.latest_bboxes = msg
        
    def eval_cb(self, msg):
        self.latest_eval = msg

    def get_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def sync_and_publish(self):
        # Check if we have the necessary data streams
        if self.current_odom is None:
            self.get_logger().warn("Waiting for Odometry...", throttle_duration_sec=2.0)
            return
            
        if self.latest_eval is None:
            self.get_logger().warn("Waiting for DWB LocalPlanEvaluation...", throttle_duration_sec=2.0)
            return
        
        # 1. Update Robot World History
        cx = self.current_odom.position.x
        cy = self.current_odom.position.y
        cyaw = self.get_yaw(self.current_odom.orientation)
        self.robot_history_world.append((cx, cy, cyaw))

        # 2. Extract Pedestrians
        dets = []
        if self.latest_bboxes is not None:
            for det_msg in self.latest_bboxes.detections:
                bx = det_msg.bbox.center.position.x
                by = det_msg.bbox.center.position.y
                sx = det_msg.bbox.size_x
                sy = det_msg.bbox.size_y
                
                x1 = bx - (sx / 2.0)
                y1 = by - (sy / 2.0)
                x2 = bx + (sx / 2.0)
                y2 = by + (sy / 2.0)

                depth = 10.0  
                if len(det_msg.results) > 0:
                    depth = det_msg.results[0].pose.pose.position.z

                dets.append({
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "depth": round(depth, 3)
                })

        self.ped_seq_buffer.append({
            "frame_idx": self.frame_counter,
            "dets": dets
        })
        self.frame_counter += 1

        self.get_logger().info(f"Buffer fill level: {len(self.robot_history_world)}/{self.Th}", throttle_duration_sec=1.0)

        # 3. Publish only when history window is full
        if len(self.robot_history_world) == self.Th:
            
            # --- A. Localize Ego History ---
            local_ego = []
            for (px, py, pyaw) in self.robot_history_world:
                dx = px - cx
                dy = py - cy
                lx = dx * math.cos(cyaw) + dy * math.sin(cyaw)
                ly = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
                local_ego.append([round(lx, 3), round(ly, 3)])

            # --- B. Process Candidates from DWB ---
            candidates = []
            expert_index = int(self.latest_eval.best_index)
            
            # Check if Nav2 is giving us global coordinates (odom/map) or local coordinates
            frame_id = self.latest_eval.header.frame_id.lower()
            is_global = 'odom' in frame_id or 'map' in frame_id

            for twist in self.latest_eval.twists:
                cand_traj = []
                for pose in twist.traj.poses:
                    px = pose.x
                    py = pose.y
                    
                    if is_global:
                        # Transform global path into the robot's current perspective
                        dx = px - cx
                        dy = py - cy
                        lx = dx * math.cos(cyaw) + dy * math.sin(cyaw)
                        ly = -dx * math.sin(cyaw) + dy * math.cos(cyaw)
                        cand_traj.append([round(lx, 3), round(ly, 3)])
                    else:
                        # Path is already local
                        cand_traj.append([round(px, 3), round(py, 3)])
                        
                # Format exactly how train.py looks for it
                candidates.append({"traj": cand_traj})

            # --- C. Build Final Unified JSON ---
            social_state = {
                "ego_history": local_ego,
                "ped_seq": list(self.ped_seq_buffer),
                "candidates": candidates,
                "expert_index": expert_index
            }

            msg = String()
            msg.data = json.dumps(social_state)
            self.state_pub.publish(msg)
            
            self.get_logger().info("✅ Publishing FULL /socialwalker_state!", throttle_duration_sec=2.0)

def main():
    rclpy.init()
    node = VisionStateInterface()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()