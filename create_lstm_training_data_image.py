# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[70]:


import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

# import score_detector
from get_keys import key_check, keys_to_output_movement, keys_to_output_action

from grab_screen import grab_screen
from score_detector import ScoreDetector
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.


from utils import label_map_util

from utils import visualization_utils as vis_util

# # Model preparation


# What model to download.
MODEL_NAME = 'fifa_graph2'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

DATA_DIRECTORY = "../../fifa_data/"
IMAGE_DIRECTORY = DATA_DIRECTORY + 'random_image_data/'
SCORE_DIRECTORY = DATA_DIRECTORY + 'random_score/'
FEATURE_DIRECTORY = DATA_DIRECTORY + 'feature_data/'
NUM_CLASSES = 3

# # ## Load a (frozen) Tensorflow model into memory.
#
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
#                                                             use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
#
#
# # ## Helper code
# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape(
#         (im_height, im_width, 3)).astype(np.uint8)


# file_name = 'rnn/training_data' + str(int(time.time())) + '.npy'

# if os.path.isfile(file_name):
#     print('File exists, loading previous data!')
#     training_data = list(np.load(file_name))
# else:
#     print('File does not exist, starting fresh!')

def collect_training_data():
    training_data = []
    training_data_all = []
    score_screen_all = []
    # count = 0
    last_keys = []
    cur_time = int(time.time())

    for i in list(range(3))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:
        image_name = IMAGE_DIRECTORY + str(cur_time)
        score_name = SCORE_DIRECTORY + str(cur_time)

        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)
            continue

        if not paused and keys:
            # keys = key_check()
            # print('keys: ' + str(keys), len(training_data))
            # if not keys:
            #     continue
            full_screen = grab_screen(region=None)
            screen = full_screen[33:753, 8:1288]
            score_screen = screen[58:93, 1126:1186]

            # image_np = cv2.resize(screen, (900, 400))
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            # image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            # (rep) = sess.run(
            #     [feature_vector],
            #     feed_dict={image_tensor: image_np_expanded})

            # save output
            if not ('space' in keys and 'space' in last_keys):
                output_movement = keys_to_output_movement(keys)
                # print("output_movement is ", output_movement)
                output_action = keys_to_output_action(keys)
                # print("output_action is", output_action)
                print('Movement:', np.argmax(output_movement), ', action:', np.argmax(output_action))
                training_data.append([screen, output_movement, output_action])
            # print([rep, output_movement, output_action])

            # only record the first shoot
            # if the game ends without shooting, the player should remark it with 'U'
            if 'space' in keys and 'space' not in last_keys or 'U' in keys:
                # print(len(training_data))
                # np.save(file_name, training_data)
                score_screen_all.append(score_screen)
                # cv2.imwrite(score_name, score_screen)
                # print('saved')
                # count += 1
                training_data_all.append(training_data)
                print("movement count is", len(training_data))
                training_data = []

            last_keys = keys
            # should save image at the end
            if 'O' in keys:
                # print('len training_data_all is ' + str(len(training_data_all)))
                print('Quitting!')
                score_screen_all.append(score_screen)

                #write score data
                print("num of score recorded is", len(score_screen_all))
                for score_index in range(len(score_screen_all)):
                    score_filename = score_name + '_' + str(score_index) + '.png'
                    cv2.imwrite(score_filename, score_screen_all[score_index])

                #write image data
                print("num of matches is %s" % len(training_data_all))
                for match_index in range(len(training_data_all)):
                    print("the num of records in match " + str(match_index) + " is " + str(len(training_data_all[match_index])) )
                    for record_index in range(len(training_data_all[match_index])):
                        image_filename = image_name + '_' + str(match_index) + '_record_' + str(record_index) \
                                        + '_movement_' + str(np.argmax(training_data_all[match_index][record_index][1])) \
                                        + '_action_' + str(np.argmax(training_data_all[match_index][record_index][2])) \
                                        + '.png'
                        cv2.imwrite(image_filename, training_data_all[match_index][record_index][0])

                # np.save(image_name, training_data_all)

                # cv2.imwrite(score_name, score_screen)
                break

# def name_score():
#     for image_file in os.listdir(IMAGE_DIRECTORY):
#
#     ingame_reward, is_success = score_detector.predict(reward_screen)
# def detect_score():
#     for filename in os.listdir(SCORE_DIRECTORY):
#         full_name = SCORE_DIRECTORY + filename

def Detection():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            feature_vector = detection_graph.get_tensor_by_name(
                "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_128/Relu6:0")

            training_data = []
            cur_score = 0
            # image_np_expanded = np.ones((1, 900, 400, 3), dtype=np.int8)

            def sort_key(x):
                arr = x.split("_")
                return (arr[0], int(arr[1]), int(arr[3]))
            image_files = os.listdir(IMAGE_DIRECTORY)
            # print(image_files)
            image_files.sort(key=lambda x:sort_key(x))
            # print(image_files)
            # the feature already dealed
            dealed_feature = set()
            for feature_file in os.listdir(FEATURE_DIRECTORY):
                feature_info = feature_file.split('_')
                if feature_info[0] in dealed_feature:
                    continue
                dealed_feature.add(feature_info[0])

            for idx, filename in enumerate(image_files):
                file_info = filename.split('_')
                fullname = IMAGE_DIRECTORY + filename
                cur_name = file_info[0] + '_' + file_info[1]
                next_name = file_info[0] + '_' + str(int(file_info[1]) + 1)
                # if os.path.isfile(FEATURE_DIRECTORY + cur_name + '.npy'):
                #     continue
                # print(filename)
                if file_info[0] in dealed_feature:
                    continue
                movement_index = int(file_info[5])
                action_index = int(file_info[7][0])
                # print(movement_index)
                # print(action_index)
                movement = np.zeros(5)
                action = np.zeros(2)
                movement[movement_index] = 1
                action[action_index] = 1
                # print(movement)
                # print(action)

                image = cv2.imread(fullname)
                # image = np.load(fullname)
                image_np = cv2.resize(image, (900, 400))
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (rep) = sess.run(
                    [feature_vector],
                    feed_dict={image_tensor: image_np_expanded})

                data = [rep, movement, action]
                training_data.append(data)

                if action[1] == 1:
                    # detect score:
                    score_detector = ScoreDetector()
                    # print(next_name)
                    score_screen = cv2.imread(SCORE_DIRECTORY + next_name + '.png' )
                    # print("score screen is ", score_screen)
                    next_score, is_success = score_detector.predict(score_screen)
                    if not is_success:
                        print("Cannot detect the score of", cur_name)
                        continue

                    this_score = next_score - cur_score
                    if this_score < 0:
                        this_score = next_score
                    np.save(FEATURE_DIRECTORY + cur_name + "_score_" + str(this_score) + '.npy', training_data)
                    print(cur_name+ "_score_" + str(this_score), len(training_data))
                    # np.save(FEATURE_DIRECTORY + cur_name + '.npy', training_data)
                    # print(cur_name, len(training_data))
                    training_data = []
                    cur_score = next_score

                # if cur_name != last_name and last_name != '0' or idx == len(image_files) - 1:
                #     np.save(FEATURE_DIRECTORY + last_name + '.npy', training_data)
                #     print(last_name, len(training_data))
                #     training_data = []
                # last_name = cur_name

                # feature0_name = FEATURE_DIRECTORY + file[:-4] + '_' + '0' + '.npy'
                # if os.path.isfile(feature0_name):
                #     print(feature0_name)
                #     continue
                # filename = IMAGE_DIRECTORY + file
                # matches = np.load(filename)
                # for i in range(len(matches)):
                #     training_data = []
                #     for record in matches[i]:
                #         image = record[0]
                #         image_np = cv2.resize(image, (900, 400))
                #         image_np_expanded = np.expand_dims(image_np, axis=0)
                #         # Actual detection.
                #         (rep) = sess.run(
                #             [feature_vector],
                #             feed_dict={image_tensor: image_np_expanded})
                #         training_data.append([rep, record[1], record[2]])
                #     np.save(FEATURE_DIRECTORY + file[:-4] + '_' + str(i), training_data)


if __name__ == '__main__':
    collect_training_data()
    # Detection()