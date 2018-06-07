#!/usr/bin/env python
import numpy as np
import pickle
import rospy
import sys

from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


if __name__ == '__main__':
    TRAINING_SET_SIZE = 50

    pick_list_name = sys.argv[1]

    rospy.init_node('capture_node')

    pick_list_1 = ['biscuits',
                   'soap',
                   'soap2']

    pick_list_2 = ['biscuits',
                   'soap',
                   'book',
                   'soap2',
                   'glue']

    pick_list_3 = ['sticky_notes',
                   'book',
                   'snacks',
                   'biscuits',
                   'eraser',
                   'soap2',
                   'soap',
                   'glue']

    if pick_list_name is "1":
        models = pick_list_1
    elif pick_list_name is "2":
        models = pick_list_2
    elif pick_list_name is "3":
        models = pick_list_3
    else:
        print("please supply pick list number: 1~3")
        sys.exit()
    print("using pick list {}".format(pick_list_name))

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        print("\ngenerating {} model...".format(model_name))
        spawn_model(model_name)

        for i in range(TRAINING_SET_SIZE):
            print("capturing {} of {}".format(i + 1, TRAINING_SET_SIZE))
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()

    pickle.dump(labeled_features, open('training_set_{}.sav'.format(pick_list_name), 'wb'))
