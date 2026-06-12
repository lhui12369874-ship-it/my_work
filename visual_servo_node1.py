#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import sys
from std_msgs.msg import String
from detection_msgs.msg import BoundingBoxes  # 注意：请确保你的YOLO输出是这个消息类型
from geometry_msgs.msg import Twist

class PIDController:
    """通用的 PID 控制器，带低通滤波和积分限幅"""
    def __init__(self, kp, ki, kd, output_limit, lpf_tau=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = output_limit
        
        self.integral = 0
        self.last_error = 0
        self.last_output = 0
        self.lpf_tau = lpf_tau  # 低通滤波时间常数
        self.last_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.033 # 防止除零，默认 30fps
        
        # 1. 积分项 (带抗饱和限幅)
        self.integral += error * dt
        self.integral = max(-self.limit/2, min(self.integral, self.limit/2))
        
        # 2. 微分项 (基于误差变化率)
        derivative = (error - self.last_error) / dt
        
        # 3. 原始 PID 输出
        raw_output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        
        # 4. 低通滤波 (消除指令毛刺，保护电机)
        alpha = dt / (self.lpf_tau + dt)
        output = self.last_output + alpha * (raw_output - self.last_output)
        
        # 5. 最终限幅
        output = max(-self.limit, min(output, self.limit))
        
        # 更新状态
        self.last_error = error
        self.last_output = output
        self.last_time = now
        return output

class AntiDroneTracker:
    def __init__(self):
        rospy.init_node('anti_drone_tracker', anonymous=True)

        # --- 动态获取终端无人机参数 ---
        self.vehicle_type = sys.argv[1] # 例如 typhoon_h480
        self.vehicle_id = sys.argv[2]   # 例如 0

        # --- 针对基于像素误差的视觉伺服参数 ---
        self.img_w = 640
        self.img_h = 360

        # --- 丢失目标保护机制相关变量 ---
        self.find_cnt = 0
        self.find_cnt_last = 0
        self.not_find_time = 0.0
        self.get_time = False
        self.cmd_str = ""
        self.twist_cmd = Twist()

        # --- 初始化 PID 控制器 ---
        # Y (左右横移)
        self.pid_y = PIDController(kp=0.15, ki=0.01, kd=0.1, output_limit=100.0)
        # Z (高度控制)
        self.pid_z = PIDController(kp=0.6, ki=0.05, kd=0.1, output_limit=100.0)
        # X (前后距离)
        self.pid_x = PIDController(kp=20.0, ki=2.0, kd=15.0, output_limit=100.0)

        # 期望目标在画面中的像素宽度
        self.target_px_width_expect = 20.0 

        # --- ROS 接口（对齐 XTDrone 命名空间） ---
        self.sub = rospy.Subscriber("/uav_"+self.vehicle_id+"/darknet_ros/bounding_boxes", BoundingBoxes, self.callback, queue_size=1)
        
        # 控制接口对齐：
        self.pub_vel = rospy.Publisher('/xtdrone/'+self.vehicle_type+'_'+self.vehicle_id+'/cmd_vel_flu', Twist, queue_size=1)
        self.pub_cmd = rospy.Publisher('/xtdrone/'+self.vehicle_type+'_'+self.vehicle_id+'/cmd', String, queue_size=1)
        
        rospy.loginfo(f"Anti-Drone Tracker Online: Tracking [person] for {self.vehicle_type}_{self.vehicle_id}")

        # 启动定时器，维持 60Hz 的稳定发送频率（与前一代码对齐）
        self.rate = rospy.Rate(60)
        self.start_loop()

    def callback(self, data):
        person_found = False
        target_box = None

        # 核心逻辑修改：遍历检测框，寻找标签为 'person' 的目标
        for box in data.bounding_boxes:
            # 注：根据你的YOLO消息类型，有可能是 box.Class == 'person' 或者 box.id == 0。此处以字符串为准
            if box.Class == 'person':
                person_found = True
                target_box = box
                break # 找到第一个人后锁定

        if person_found:
            self.find_cnt += 1
            self.get_time = False
            self.cmd_str = "" # 发现目标，清空HOVER等状态命令

            bw = target_box.xmax - target_box.xmin
            bh = target_box.ymax - target_box.ymin
            cx = target_box.xmin + bw / 2.0
            cy = target_box.ymin + bh / 2.0

            # 计算误差
            err_y = (self.img_w / 2.0) - cx  
            err_z = (self.img_h / 2.0) - cy 
            err_x = self.target_px_width_expect - bw 

            # PID 解算
            v_y = self.pid_y.update(err_y)
            v_z = self.pid_z.update(err_z)
            v_x = self.pid_x.update(err_x)

            # 填充 Twist 速度消息 (FLU机体系)
            self.twist_cmd.linear.x = v_x / 50.0  # 前后
            self.twist_cmd.linear.y = v_y / 50.0  # 左右
            self.twist_cmd.linear.z = v_z / 50.0  # 上下
            self.twist_cmd.angular.x = 0.0
            self.twist_cmd.angular.y = 0.0
            self.twist_cmd.angular.z = 0.0  

            rospy.loginfo_throttle(1.0, f"Tracking Human | vX:{self.twist_cmd.linear.x:.2f} | vY:{self.twist_cmd.linear.y:.2f} | vZ:{self.twist_cmd.linear.z:.2f}")

    def start_loop(self):
        """主控制循环，负责定时发布和丢失保护"""
        while not rospy.is_shutdown():
            # 1. 周期性发布当前的速度和指令
            self.pub_vel.publish(self.twist_cmd)
            self.pub_cmd.publish(String(data=self.cmd_str))

            # 2. 丢失目标保护机制（对齐第一段代码）
            if self.find_cnt - self.find_cnt_last == 0:
                if not self.get_time:
                    self.not_find_time = rospy.get_time()
                    self.get_time = True
                
                # 如果超过 2 秒没看到人
                if rospy.get_time() - self.not_find_time > 2.0:
                    self.twist_cmd.linear.x = 0.0
                    self.twist_cmd.linear.y = 0.0
                    self.twist_cmd.linear.z = 0.0
                    self.cmd_str = 'HOVER'  # 触发 XTDrone 悬停命令
                    rospy.logwarn_throttle(2.0, "Target Lost! Hovering...")
                    self.get_time = False

            self.find_cnt_last = self.find_cnt
            self.rate.sleep()

if __name__ == '__main__':
    try:
        # 防呆检查参数
        if len(sys.argv) < 3:
            print("Please provide vehicle_type and vehicle_id. Example: python script.py typhoon_h480 0")
            sys.exit(1)
        tracker = AntiDroneTracker()
    except rospy.ROSInterruptException:
        pass