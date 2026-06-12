#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from detection_msgs.msg import BoundingBoxes
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

        # --- 针对基于像素误差的视觉伺服参数 ---
        self.img_w = 752
        self.img_h = 480
        self.detection_state = 0

        # --- 初始化 PID 控制器 ---
        # Y (左右横移): 横移需要机身侧倾，惯性较大。增大了 Kd 来加强刹车效果，防止左右震荡 (钟摆效应)
        self.pid_y = PIDController(kp=0.15, ki=0.01, kd=0.1, output_limit=100.0)
        
        # Z (高度控制): 相对平稳
        self.pid_z = PIDController(kp=0.6, ki=0.05, kd=0.1, output_limit=100.0)
        
        # X (前后距离): 室外追逐需要较强的 D 项防止前冲
        self.pid_x = PIDController(kp=50.0, ki=2.0, kd=15.0, output_limit=100.0)

        # 期望目标在画面中的像素宽度 (室外建议保持远一点，安全第一)
        self.target_px_width_expect = 20.0 

        # ROS 接口
        self.sub = rospy.Subscriber("/yolov5/detections", BoundingBoxes, self.callback)
        self.pub_vel = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=1)
        
        rospy.loginfo("Anti-Drone Tracker Online: Outdoor Strafe (Y-axis translation) Mode")

    def callback(self, data):
        if len(data.bounding_boxes) > 0:
            self.detection_state = 1
            
            box = data.bounding_boxes[0]
            bw = box.xmax - box.xmin
            bh = box.ymax - box.ymin
            cx = box.xmin + bw / 2.0
            cy = box.ymin + bh / 2.0

            # 2. 计算误差
            # Y轴（横移）误差：
            # 目标在左侧（cx较小），(img_w/2 - cx) 为正。ROS中Y速度为正代表向左横移，符合预期。
            err_y = (self.img_w / 2.0) - cx  
            
            # 高度误差 (像素)
            err_z = (self.img_h / 2.0) - cy 
            
            # 距离误差 (像素宽度误差)
            err_x = self.target_px_width_expect - bw 

            # 3. PID 解算
            v_y = self.pid_y.update(err_y)
            v_z = self.pid_z.update(err_z)
            v_x = self.pid_x.update(err_x)

            # 4. 发布指令
            cmd = Twist()
            # 线速度 (平移)
            cmd.linear.x = v_x / 50.0  # 前后距离控制 (m/s)
            cmd.linear.y = v_y / 50.0  # 左右横移控制 (m/s)
            cmd.linear.z = v_z / 50.0  # 高度升降控制 (m/s)

            # 角速度 (旋转)：全部设为 0，云台/飞控接管朝向锁定
            cmd.angular.x = 0.0
            cmd.angular.y = 0.0
            cmd.angular.z = 0.0  

            self.pub_vel.publish(cmd)

            # 修复了原来 loginfo 打印报错的 bug
            rospy.loginfo(f"Tracking: {box.Class} | vX:{cmd.linear.x:.2f}m/s | vY:{cmd.linear.y:.2f}m/s | vZ:{cmd.linear.z:.2f}m/s")
        else:
            self.detection_state = 0
            # 丢失目标保护：发送全为0的指令，无人机会在当前位置悬停急刹
            self.pub_vel.publish(Twist())

if __name__ == '__main__':
    try:
        tracker = AntiDroneTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
