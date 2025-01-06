#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from filterpy.kalman import KalmanFilter
import numpy as np
from math import radians
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag

def angle_from_to_2(angle, min_angle, max_angle):
    while angle < min_angle:
        angle += 360
    while angle >= max_angle:
        angle -= 360
    return angle

class KalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_node')
        self.callback_called = False
        # Inicialize o filtro de Kalman
        # self.kf = KalmanFilter(dim_x=12, dim_z=6)
        self.P = np.zeros((12,12))
        self.P_pred = np.zeros((12,12))
        # Inicialize a matriz de transição de estado (A)
        self.F = np.eye(12)        
        # Inicialize a matriz de observação (H)
        Matriz_H = np.zeros((6, 12))
        Matriz_H[0,0] = 1
        Matriz_H[1,4] = 1
        Matriz_H[2,8] = 1
        Matriz_H[3,6] = 1
        Matriz_H[4,2] = 1
        Matriz_H[5,10] = 1
        self.H = Matriz_H     # Matriz de transformação de medida
        
        # Defina a covariância do processo (Q) e a covariância da medida (R)

        # x_theta_matrix = Q_discrete_white_noise(dim=4, dt=1, var=0.1) + Q_discrete_white_noise(dim=4, dt=1, var=0.5)
        # y_roll_matrix = Q_discrete_white_noise(dim=4, dt=1, var=0.1) + Q_discrete_white_noise(dim=4, dt=1, var=0.5)
        # psi_matrix = Q_discrete_white_noise(dim=2, dt=1, var=0.1)
        # z_matrix = Q_discrete_white_noise(dim=2, dt=1, var=0.1)   
        # self.kf.Q = block_diag(x_theta_matrix, y_roll_matrix,psi_matrix,z_matrix)  # Matriz de covariância do processo
        self.x_pred = np.zeros(12)


        # self.z  = np.zeros((1,6))
        self.z = np.array([
            0,
            0,
            0,
            0,
            0,
            0,
            # np.radians(angle_from_to_2(np.degrees(euler_angles[2]),-180,180)),
        ])
        self.Q = np.diag([0.2] * 12)
        self.R = np.diag([1] * 6)
        # self.kf.R = np.diag([1.0,0.5,0.5,0.5,0.5,0.5]) # Matriz de covariância da medida 
        print(self.Q)
        print(self.R)
        # Subscribers
        rospy.Subscriber('/robot_aruco_pose', PoseStamped, self.aruco_pose_callback)
        # rospy.Subscriber('/bebop2/odometry_sensor1/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/bebop2/odometry_sensor1/odometry', Odometry, self.odometry_callback)
        # rospy.Subscriber('/bebop2/ground_truth/odometry', Odometry, self.odometry_callback)
        # Publisher
        self.filtered_pose_pub = rospy.Publisher('/filtered_pose', Odometry, queue_size=10)

    def verificar_callback(self):
        if self.callback_called:
            callback_called_key = True
        else:
            callback_called_key = False
        self.callback_called = False
        return callback_called_key
    
    def aruco_pose_callback(self, aruco_pose):
        quaternion = [aruco_pose.pose.orientation.x, aruco_pose.pose.orientation.y, aruco_pose.pose.orientation.z, aruco_pose.pose.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)        
        self.z = np.array([
            aruco_pose.pose.position.x,
            aruco_pose.pose.position.y,
            aruco_pose.pose.position.z,
            euler_angles[0],
            euler_angles[1],
            euler_angles[2],
            # np.radians(angle_from_to_2(np.degrees(euler_angles[2]),-180,180)),
        ])
        # print(self.z)
        self.callback_called = True      

    def odometry_callback(self, odometry):
        # Atualize o modelo de predição com as informações de odometria
        quaternion = [odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)
        x      = odometry.pose.pose.position.x
        dx     = odometry.twist.twist.linear.x
        theta  = euler_angles[1] #rad
        dtheta = odometry.twist.twist.angular.y #rad/s
        y      = odometry.pose.pose.position.y
        dy     = odometry.twist.twist.linear.y
        roll   = euler_angles[0] #rad
        droll  = odometry.twist.twist.angular.x #rad/s
        z      = odometry.pose.pose.position.z
        dz     = odometry.twist.twist.linear.z
        yaw    = np.radians(angle_from_to_2(np.degrees(euler_angles[2]),-180,180)) #rad
        # yaw    = euler_angles[2]
        dyaw   = odometry.twist.twist.angular.z
        
        self.x_odom = np.array([x,dx,theta,dtheta,y,dy,roll,droll,z,dz,yaw,dyaw]).T
        self.x_pred = np.dot(self.F, self.x_odom)
        self.P_pred = np.dot(self.F, np.dot(self.P,self.F.T)) + self.Q
        
    def run(self):
        rate = rospy.Rate(100)  # Taxa de execução em Hz
        while not rospy.is_shutdown():
            rate.sleep()
            
            # if self.verificar_callback():
                # Update
            self.S = np.dot(self.H, np.dot(self.P_pred,self.H.T)) + self.R
            self.K = np.dot(self.P_pred, np.dot(self.H.T,np.linalg.inv(self.S)))
            self.Y = self.z - np.dot(self.H,self.x_pred)
            self.x = self.x_pred + np.dot(self.K,self.Y)
            aux      = (np.eye(12) - np.dot(self.K,self.H))
            self.P = np.dot(aux,np.dot(self.P_pred,aux.T)) + np.dot(self.K,np.dot(self.R,self.K.T))
            # print(self.x_pred )
            # else:
            #     self.x = self.x_pred

            # Publicar a pose filtrada            
            filtered_pose = Odometry()
            filtered_pose.header.stamp = rospy.Time.now()
            filtered_pose.header.frame_id = "world"
            filtered_pose.pose.pose.position.x = self.x[0]
            filtered_pose.pose.pose.position.y = self.x[4]
            filtered_pose.pose.pose.position.z = self.x[8]
            euler_angles = [self.x[6], self.x[2], self.x[10]]
            quaternion = quaternion_from_euler(euler_angles[0],euler_angles[1],np.radians(angle_from_to_2(np.degrees(euler_angles[2]),-180,180)))
            filtered_pose.pose.pose.orientation.x = quaternion[0]
            filtered_pose.pose.pose.orientation.y = quaternion[1]
            filtered_pose.pose.pose.orientation.z = quaternion[2]
            filtered_pose.pose.pose.orientation.w = quaternion[3]
            filtered_pose.twist.twist.linear.x = self.x[1]
            filtered_pose.twist.twist.linear.y = self.x[5]
            filtered_pose.twist.twist.linear.z = self.x[9]
            filtered_pose.twist.twist.angular.x = self.x[7]
            filtered_pose.twist.twist.angular.y = self.x[3]
            filtered_pose.twist.twist.angular.z = self.x[11]
            # print(filtered_pose)
            self.filtered_pose_pub.publish(filtered_pose)
            
if __name__ == '__main__':
    try:
        kalman_filter_node = KalmanFilterNode()
        kalman_filter_node.run()
    except rospy.ROSInterruptException:
        pass
