#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from filterpy.kalman import KalmanFilter
import numpy as np
from math import radians
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


class KalmanFilterNode:
    def __init__(self):
        rospy.init_node('kalman_filter_node')
        self.callback_called = False
        # Inicialize o filtro de Kalman
        self.kf = KalmanFilter(dim_x=12, dim_z=6)

        # Inicialize a matriz de transição de estado (A)
        self.kf.F = np.eye(12)
        
        # Inicialize a matriz de observação (H)
        Matriz_H = np.zeros((6, 12))
        Matriz_H[0,0] = 1
        Matriz_H[1,1] = 1
        Matriz_H[2,2] = 1
        Matriz_H[3,6] = 1
        Matriz_H[4,7] = 1
        Matriz_H[5,8] = 1
        self.kf.H = Matriz_H     # Matriz de transformação de medida
        
        # Defina a covariância do processo (Q) e a covariância da medida (R)

        # x_theta_matrix = Q_discrete_white_noise(dim=4, dt=1, var=0.1) + Q_discrete_white_noise(dim=4, dt=1, var=0.5)
        # y_roll_matrix = Q_discrete_white_noise(dim=4, dt=1, var=0.1) + Q_discrete_white_noise(dim=4, dt=1, var=0.5)
        # psi_matrix = Q_discrete_white_noise(dim=2, dt=1, var=0.1)
        # z_matrix = Q_discrete_white_noise(dim=2, dt=1, var=0.1)   
        # self.kf.Q = block_diag(x_theta_matrix, y_roll_matrix,psi_matrix,z_matrix)  # Matriz de covariância do processo
        
        self.kf.Q = np.diag([0.2] * 12)
        self.kf.R = np.diag([1] * 6)
        # self.kf.R = np.diag([1.0,0.5,0.5,0.5,0.5,0.5]) # Matriz de covariância da medida 
        print(self.kf.Q)
        print(self.kf.R)
        # Subscribers
        rospy.Subscriber('/robot_aruco_pose', PoseStamped, self.aruco_pose_callback)
        # rospy.Subscriber('/bebop2/odometry_sensor1/odometry', Odometry, self.odometry_callback)
        rospy.Subscriber('/bebop2/odometry_sensor1/odometry', Odometry, self.odometry_callback)
        # Publisher
        self.filtered_pose_pub = rospy.Publisher('/filtered_pose', PoseStamped, queue_size=10)

    def aruco_pose_callback(self, aruco_pose):
        # Atualize o filtro com as informações da pose do marcador ArUco
        quaternion = [aruco_pose.pose.orientation.x, aruco_pose.pose.orientation.y, aruco_pose.pose.orientation.z, aruco_pose.pose.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)
        
        z = np.array([
            aruco_pose.pose.position.x,
            aruco_pose.pose.position.y,
            aruco_pose.pose.position.z,
            euler_angles[0],
            euler_angles[1],
            euler_angles[2],
        ])
        self.callback_called = True
        print(z)
        self.kf.update(z)



    def odometry_callback(self, odometry):
        # Atualize o modelo de predição com as informações de odometria
        quaternion = [odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)
        self.kf.F[0, 0] = odometry.pose.pose.position.x
        self.kf.F[1, 1] = odometry.pose.pose.position.y
        self.kf.F[2, 2] = odometry.pose.pose.position.z
        self.kf.F[3, 3] = odometry.twist.twist.linear.x
        self.kf.F[4, 4] = odometry.twist.twist.linear.y
        self.kf.F[5, 5] = odometry.twist.twist.linear.z
        self.kf.F[6, 6] = euler_angles[0]
        self.kf.F[7, 7] = euler_angles[1]
        self.kf.F[8, 8] = euler_angles[2]
        self.kf.F[9, 9] = odometry.twist.twist.angular.x
        self.kf.F[10, 10] = odometry.twist.twist.angular.y
        self.kf.F[11, 11] = odometry.twist.twist.angular.z

        self.kf.predict()
    def verificar_callback(self):
        if self.callback_called:
            print("Callback foi chamado!")
        else:
            print("Callback não foi chamado.")
        self.callback_called = False
    def run(self):
        rate = rospy.Rate(5)  # Taxa de execução em Hz
        while not rospy.is_shutdown():
            rate.sleep()
            # Publicar a pose filtrada
            filtered_pose = PoseStamped()
            filtered_pose.header.stamp = rospy.Time.now()
            filtered_pose.pose.position.x = self.kf.x_post[0]
            filtered_pose.pose.position.y = self.kf.x_post[1]
            filtered_pose.pose.position.z = self.kf.x_post[2]
            # Converta as orientações de radianos para graus para publicação
            euler_angles = [self.kf.x_post[6], self.kf.x_post[7], self.kf.x_post[8]]
            # filtered_pose = PoseStamped()
            # filtered_pose.header.stamp = rospy.Time.now()
            # filtered_pose.pose.position.x = self.kf.x[0]
            # filtered_pose.pose.position.y = self.kf.x[1]
            # filtered_pose.pose.position.z = self.kf.x[2]
            # # Converta as orientações de radianos para graus para publicação
            # euler_angles = [self.kf.x[6], self.kf.x[7], self.kf.x[8]]
            quaternion = quaternion_from_euler(euler_angles[0],euler_angles[1],euler_angles[2])
            filtered_pose.pose.orientation.x = quaternion[0]
            filtered_pose.pose.orientation.y = quaternion[1]
            filtered_pose.pose.orientation.z = quaternion[2]
            filtered_pose.pose.orientation.w = quaternion[3]
            # print(filtered_pose)
            self.filtered_pose_pub.publish(filtered_pose)
            self.verificar_callback()
            
if __name__ == '__main__':
    try:
        kalman_filter_node = KalmanFilterNode()
        kalman_filter_node.run()
    except rospy.ROSInterruptException:
        pass
