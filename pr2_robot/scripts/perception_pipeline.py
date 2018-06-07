#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml
import xml.etree.ElementTree as ET


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"] = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Test
    test_points_pub.publish(pcl_msg)

    # Exercise-2 TODOs:
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    ###########################################################################
    # 1. Voxel Grid Downsampling
    ###########################################################################
    # Create a VoxelGrid filter object for our input point cloud
    vox = cloud.make_voxel_grid_filter()

    # Choose a voxel (also known as leaf) size
    # Note: this (1) is a poor choice of leaf size
    # Experiment and find the appropriate size!
    LEAF_SIZE = 0.01

    # Set the voxel (or leaf) size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    ###########################################################################
    # 2. PassThrough Filter
    ###########################################################################
    # Create a PassThrough filter object for z axis
    passthrough_z = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough_z.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough_z.filter()

    # Create a PassThrough filter object for y axis
    passthrough_y = cloud_filtered.make_passthrough_filter()

    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough_y.set_filter_field_name(filter_axis)
    axis_min = -0.55
    axis_max = 0.55
    passthrough_y.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud.
    cloud_filtered = passthrough_y.filter()

    ###########################################################################
    # 3. Outlier removal filter
    ###########################################################################
    outlier_filter = cloud_filtered.make_statistical_outlier_filter()

    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(50)

    # Set threshold scale factor
    x = 1.0

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()

    ###########################################################################
    # 4. RANSAC Plane Segmentation
    ###########################################################################
    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance
    # for segmenting the table
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    ###########################################################################
    # 5. Extract inliers and outliers
    ###########################################################################
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)  # table
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)  # objects on table

    ###########################################################################
    # 6. Euclidean Clustering
    ###########################################################################
    # Apply function to convert XYZRGB to XYZ
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    # as well as minimum and maximum cluster size (in points)
    # NOTE: These are poor choices of clustering parameters
    # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(10000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    ###########################################################################
    # 7. Create Cluster-Mask Point Cloud to visualize each cluster separately
    ###########################################################################
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    ###########################################################################
    # 8. Create new cloud containing all clusters, each with unique color
    ###########################################################################
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    ###########################################################################
    # 9. Convert PCL data to ROS messages
    ###########################################################################
    ros_objects_cloud = pcl_to_ros(extracted_outliers)
    ros_table_cloud = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    ###########################################################################
    # 10. Publish ROS messages
    ###########################################################################
    pcl_objects_pub.publish(ros_objects_cloud)
    pcl_table_pub.publish(ros_table_cloud)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Exercise-3 TODOs:

    ###########################################################################
    # 11. Classify the clusters (loop through each detected cluster one at a time)
    ###########################################################################
    detected_objects_labels = []
    detected_objects = []
    cloud_objects = extracted_outliers

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)

        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    ###########################################################################
    # 12. Publish the list of detected objects
    ###########################################################################
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    ###########################################################################
    # 13. Call mover to pick up object
    ###########################################################################
    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover(detected_objects):
    labels = []
    centroids = []  # to be list of tuples (x, y, z)
    pick_place_msg_list = []
    dict_list = []  # output dict
    for do in detected_objects:
        ###########################################################################
        # 1. Get detected object cloud info
        ###########################################################################
        do_label = str(do.label)
        labels.append(do_label)
        points_arr = ros_to_pcl(do.cloud).to_array()
        # ROS messages expect "float" but np.mean gives "numpy.float64". Use np.asscalar() to recast back to "float".
        do_centroid = np.mean(points_arr, axis=0)[:3]
        centroids.append(do_centroid)

        ###########################################################################
        # 2. Get group info
        ###########################################################################
        do_group = filter(lambda x: do_label == x['name'], object_list_param)[0]['group']

        ###########################################################################
        # 3. Get target dropbox info
        ###########################################################################
        do_dropbox_name, do_dropbox_position = get_dropbox_by_group(do_group)

        ###########################################################################
        # 4. Convert data type
        ##########################################################################
        test_scene_num = Int32()
        test_scene_num.data = test_scenario_idx

        object_name = String()
        object_name.data = do_label

        arm_name = String()
        arm_name.data = do_dropbox_name

        pick_pose = Pose()
        pick_pose.position.x = np.asscalar(do_centroid[0])
        pick_pose.position.y = np.asscalar(do_centroid[1])
        pick_pose.position.z = np.asscalar(do_centroid[2])

        place_pose = Pose()
        place_pose.position.x = do_dropbox_position[0]
        place_pose.position.y = do_dropbox_position[1]
        place_pose.position.z = do_dropbox_position[2]

        pick_place_msg_list.append((test_scene_num, object_name, arm_name, pick_pose, place_pose))

        ###########################################################################
        # 5. Generate dict info
        ###########################################################################
        yaml_dict = make_yaml_dict(
            test_scene_num=test_scene_num,
            arm_name=arm_name,
            object_name=object_name,
            pick_pose=pick_pose,
            place_pose=place_pose)
        dict_list.append(yaml_dict)

    ###########################################################################
    # 6. Save dict info to yaml file
    ###########################################################################
    yaml_filename = os.path.join(PROJECT_ROOT_DIR, "output_{}.yaml".format(pick_up_list_idx))

    try:
        send_to_yaml(yaml_filename, dict_list)
        print("output yaml file saved to {}".format(yaml_filename))
    except:
        print("failed to save output yaml file to disk!")

    ###########################################################################
    # 7. Send msg to pick_place_routine
    ##########################################################################
    for test_scene_num, object_name, arm_name, pick_pose, place_pose in pick_place_msg_list:
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            resp = pick_place_routine(
                test_scene_num=test_scene_num,
                object_name=object_name,
                arm_name=arm_name,
                pick_pose=pick_pose,
                place_pose=place_pose)
            print("pickup msg sent for {}".format(object_name.data))
            print("response: ", resp.success)

        except rospy.ServiceException, e:
            print("service call failed: {}".format(e))


def get_dropbox_by_group(group):
    dropbox = filter(lambda x: group == x['group'], dropbox_param)[0]
    return dropbox['name'], dropbox['position']


if __name__ == '__main__':
    # Directories
    CURRENT_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT_DIR = os.path.join(CURRENT_FILE_DIR, '../../')

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    test_points_pub = rospy.Publisher("/test_points", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # Extract info from launch file
    tree = ET.parse(os.path.join(CURRENT_FILE_DIR, '../launch/pick_place_project.launch'))
    root = tree.getroot()

    test_scenario_file_name = list(filter(lambda x: 'test' in x.get('value'), list(root.iter('arg'))))[0].get('value')
    test_scenario_idx = int(test_scenario_file_name[-7])
    pick_up_list_file_name = list(filter(lambda x: 'pick_list' in x.get('file'), root.findall('rosparam')))[0].get(
        'file')
    pick_up_list_idx = int(pick_up_list_file_name[-6])
    print("loading scenario file: {}".format(test_scenario_file_name))
    print("loading pick up list file: {}".format(pick_up_list_file_name))

    # Load Model From disk
    # Use "model_1.save", "model_2.sav" and "model_3.sav" for each corresponding pick list defined in
    # pick_place_object.launch.
    # Assume model stored in project root folder
    trained_model_file_name = 'model_{}.sav'.format(pick_up_list_idx)
    model = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, trained_model_file_name), 'rb'))
    print("loading trained model file: {}".format(trained_model_file_name))

    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Load object list in pick up file
    object_list_param = rospy.get_param('/object_list')

    # Load dropbox info
    dropbox_param = rospy.get_param('/dropbox')

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
