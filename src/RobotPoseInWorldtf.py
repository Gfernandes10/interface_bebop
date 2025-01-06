#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from tf.transformations import rotation_matrix, quaternion_from_matrix, translation_matrix, quaternion_matrix, quaternion_from_euler,euler_from_quaternion, euler_from_matrix
def angle_from_to_2(angle, min_angle, max_angle):
    while angle < min_angle:
        angle += 360
    while angle >= max_angle:
        angle -= 360
    return angle
class RobotPoseToWorld:
    def __init__(self):
        rospy.init_node('Bebop_pose_word', anonymous=False)

        # Subscribers
        rospy.Subscriber('/aruco_single/pose', PoseStamped, self.robot_pose_rel_tag_callback)

        # Publisher
        self.robot_pose_world_pub = rospy.Publisher('/robot_aruco_pose', PoseStamped, queue_size=10)

        # Pose tag word
        TagPx = rospy.get_param('~TagPx', 0.7)
        TagPy = rospy.get_param('~TagPy', 0.0)
        TagPz = rospy.get_param('~TagPz', 1.0)
        TagRoll = rospy.get_param('~TagRoll', 0.0)
        TagPitch = rospy.get_param('~TagPitch', -1.57)
        TagYaw = rospy.get_param('~TagYaw', 0.0)
        print(TagRoll)
        print(TagPitch)
        print(TagYaw)
        Tagquaternion = quaternion_from_euler(TagRoll, TagPitch, TagYaw)
        TagOx, TagOy, TagOz, TagOw = Tagquaternion
        translation = translation_matrix([TagPx, TagPy, TagPz])
        rotation = quaternion_matrix([TagOx, TagOy, TagOz, TagOw])

        self.T_tag_world = np.dot(translation, rotation)

        
        # Transformation matrices
        PoseTag = transform_matrix_to_pose(self.T_tag_world)
        self.T_robot_tag = None

        
        print("Matriz")
        print(self.T_tag_world)
        print("função")
        print(pose_to_transform_matrix(PoseTag.pose))
        print("pose")
        print(PoseTag)
        euler_rotation = euler_from_quaternion([PoseTag.pose.orientation.x, PoseTag.pose.orientation.y, PoseTag.pose.orientation.z, PoseTag.pose.orientation.w,'sxyz'])
        print("Euler")
        print(euler_rotation)
        # Parâmetro para controlar a suavização
        # self.smoothing_factor = rospy.get_param('~smoothing_factor', 1.5)

        # Armazenamento de leituras anteriores
        self.previous_poses = []

    def robot_pose_rel_tag_callback(self, robot_pose_rel_tag):
        # Transform robot pose to transformation matrix
        self.T_robot_tag = pose_to_transform_matrix(robot_pose_rel_tag.pose)

        # Se ambas as transformações (robot-to-tag e tag-to-world) estiverem disponíveis
        if self.T_robot_tag is not None:
            # Componha as transformações
            inv_T_robot_tag = np.linalg.inv(self.T_robot_tag)
            T_robot_world = np.dot(self.T_tag_world, inv_T_robot_tag)

            # Aplicar suavização com média móvel
            self.previous_poses.append(transform_matrix_to_pose(T_robot_world))
            if len(self.previous_poses) > 4:
                self.previous_poses.pop(0)

            smoothed_pose = self.smooth_poses()

            # Publish a pose do robô no sistema de coordenadas global
            self.robot_pose_world_pub.publish(smoothed_pose)
        


    def smooth_poses(self):
        avg_pose = PoseStamped()
        avg_pose.header.stamp = rospy.Time.now()

        num_poses = len(self.previous_poses)
        if num_poses > 0:
            avg_translation = np.zeros(3)
            avg_rotation = np.zeros(4)

            for pose in self.previous_poses:
                avg_translation += np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
                avg_rotation += np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])

            avg_translation /= num_poses
            avg_rotation /= num_poses

            avg_pose.pose.position.x = avg_translation[0]
            avg_pose.pose.position.y = avg_translation[1]
            avg_pose.pose.position.z = avg_translation[2]
            avg_rotation_euler = euler_from_quaternion(avg_rotation)
            avg_rotation_euler_fix = [avg_rotation_euler[0], avg_rotation_euler[1], np.radians(angle_from_to_2(np.degrees(avg_rotation_euler[2]),-180,180))]
            avg_rotation = quaternion_from_euler(avg_rotation_euler_fix[0],avg_rotation_euler_fix[1],avg_rotation_euler_fix[2])
            avg_pose.pose.orientation.x = avg_rotation[0]
            avg_pose.pose.orientation.y = avg_rotation[1]
            avg_pose.pose.orientation.z = avg_rotation[2]
            avg_pose.pose.orientation.w = avg_rotation[3]

        return avg_pose


def pose_to_transform_matrix(pose):
    translation = translation_matrix([pose.position.x, pose.position.y, pose.position.z])
    rotation = quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return np.dot(translation, rotation)


def transform_matrix_to_pose(transform_matrix):
    pose = PoseStamped()

    translation = transform_matrix[0:3, 3]

    pose.pose.position.x = translation[0]
    pose.pose.position.y = translation[1]
    pose.pose.position.z = translation[2]

    rotation_matrix = transform_matrix[0:4, 0:4]

    rotation = quaternion_from_matrix(rotation_matrix)

    pose.header.stamp = rospy.Time.now()

    pose.pose.orientation.x = rotation[0]
    pose.pose.orientation.y = rotation[1]
    pose.pose.orientation.z = rotation[2]
    pose.pose.orientation.w = rotation[3]

    return pose


if __name__ == '__main__':
    try:
        robot_pose_to_world_node = RobotPoseToWorld()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
