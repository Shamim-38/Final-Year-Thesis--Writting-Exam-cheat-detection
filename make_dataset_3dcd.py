"""
Author: Md Mahedi Hasan
Description: Preprocess pose sequence dataset to feed rnn model
Steps to do
        1. find out and sort partial body
        2. normalize keypoints
        3. handle no person, multiple person
        4. make train & validation dataset
"""

# python packages
import os
import json, glob
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.utils import to_categorical


# project modules
from .. import config
from . import hand_features_3dcd as hf

# for motion features
first_frame_bkps = []

# formating json file
def handling_json_data_file(data):
    global first_frame_bkps
    combined_features = []
    is_no_people = False
    is_partial_body = False
    
    # no people detected
    if len(data["people"]) == 0:
        is_no_people = True

    # one people detected 
    else:
        pose_keypoints = data["people"][0]["pose_keypoints_2d"]
        is_partial_body =  hf.is_partial_body(pose_keypoints)

        # for complete pose
        if(not is_partial_body):
            pose_features = hf.normalize_keypoints(pose_keypoints)

            """
            limb_features = hf.get_body_limb(pose_keypoints)
            angle_features = hf.get_joint_angle(pose_keypoints)
            
            # for first frame, store the  bpks and skip the motion feat.
            if(len(first_frame_bkps) == 0):
                first_frame_bkps = pose_keypoints
                is_no_people = True

            else:
                second_frame_bpks = pose_keypoints
                motion_features = hf.get_motion_featurs(second_frame_bpks, 
                                                  first_frame_bkps)
                first_frame_bkps = second_frame_bpks
            """
    
    # combining all fetures
    if (not is_partial_body and not is_no_people):
        combined_features = pose_features
    """
        combined_features += limb_features
        combined_features += angle_features
        combined_features += motion_features
    """
    return combined_features, is_no_people, is_partial_body




# dataset formatted for rnn input
def get_format_data(subject_id,
                    seq_kps,
                    seq,
                    start_id):

    seq_data = []
    seq_label = []
    
    # check how many image frame of length 28 we can get
    nb_images = len(seq_kps)

    # for larger than 15 image sequene creating one timestep
    if(nb_images < config.casiaB_nb_steps):
        if ((config.casiaB_nb_steps - nb_images) > (config.casiaB_nb_steps / 2)):
            nb_image_set = 0

        else:
            nb_image_set = 1
            seq_kps = seq_kps * 2
        
    else:
        nb_image_set = int((nb_images - config.casiaB_nb_steps) / 
                            config.actual_fps) + 1

    # finding label of from subject data file
    sub_label = int(subject_id[1:]) - start_id
    print(seq, "has total image:", nb_images, 
            "  total image_set:", nb_image_set)

    # for some value of image_set
    if(nb_image_set > 0):
        for i in range(0, nb_image_set):
            start_frame_id = i * config.actual_fps
            end_frame_id = start_frame_id + config.casiaB_nb_steps

            # saving each keypoints
            for line in range(start_frame_id, end_frame_id):
                seq_data.append(seq_kps[line])
                seq_label.append([sub_label])

        seq_data = np.array(seq_data)
        seq_label = np.array(seq_label)

        seq_data = np.array(np.split(seq_data, nb_image_set))
        seq_label = np.array(np.split(seq_label, nb_image_set))

    return seq_data, seq_label



def get_keypoints_for_all_cheat(cheat_type_list):

    print("\n\n*********** Generating %s data ***********" % "training")    
    total_dataset = []
    total_dataset_label = []

    for cheat_type in cheat_type_list[:1]:
        print("\n\n\n\n############ cheat type %s ############" % cheat_type)

        # variable for each subject
        cheat_label = []
        
        cheat_total_frame = 0
        cheat_total_no_people = 0
        cheat_total_partial_body = 0

        # getting angle
        cheat_dir = os.path.join(config.pose_3dcd_path(), cheat_type)
        cheat_vid_list = os.listdir(cheat_dir)
        #print(cheat_vid_list)

        num_cheat_vid =  len(cheat_vid_list)
        print("%s has: %d cheat vidoes" % (cheat_type, num_cheat_vid))


        # considering each cheat video
        for cheat_vid in cheat_vid_list:
            cheat_vid_dir = os.path.join(cheat_dir, cheat_vid)

            cheat_vid_data = []
            cheat_vid_label = []
            
            # considering each cheat vids
            os.chdir(cheat_vid_dir)

            # getting all json files
            json_files = sorted(glob.glob("*.json"))
            cheat_total_frame += len(json_files)
                                
            for row, file in enumerate(json_files):
                with open(file) as data_file:
                    data = json.load(data_file)
                    
                    frame_kps, no_people, partial_body = handling_json_data_file(data)
                    #print("frame no: ", f+1); print(frame_kps)
        
                    # counting no, multiple people and partial body detected
                    if (no_people == True):  cheat_total_no_people += 1
                    elif (partial_body == True): cheat_total_partial_body += 1
                        
                    # for single people save the frame key points
                    else:
                        cheat_vid_data.append(frame_kps)
            
            print(len(cheat_vid_data))
            """
            # saving each seq walking data
            seq_data, seq_label = get_format_data(subject_id,
                                                seq_kps,
                                                seq,
                                                start_id)
            
            # adding each angle all seq data and label except empty list
            if(seq_data != []):
                angle_data_list.append(seq_data)
                angle_label_list.append(seq_label)

        # saving each angle walking data except empty list
        if (len(angle_data_list) != 0):
            angle_data = np.vstack(angle_data_list)
            angle_label = np.vstack(angle_label_list)

            print("angle data shape:", angle_data.shape)
            print("angle label shape:", angle_label.shape)

            sub_data.append(angle_data)
            
            # convert it to categorical value
            sub_label.append(to_categorical(angle_label, 
                            config.casiaB_nb_classes))

        # collecting all subject data
        total_dataset.append(sub_data)
        total_dataset_label.append(sub_label)
        
        # per subject display info
        print("\nsubject id", subject_id, "has total image set:", 
                                    sum(len(i) for i in sub_data))
        
        print("total frame:", sub_total_frame)
        sub_single_people = sub_total_frame - (sub_total_no_people +
                                               sub_total_partial_body)
        
        print("suitable frame for detection:", sub_single_people)
        print("no people detected:", sub_total_no_people)
        print("partial people detected:", sub_total_partial_body)
        #### end of each subject work

    return total_dataset, total_dataset_label
        """


if __name__ == "__main__":
    cheat_type_list = os.listdir(config.pose_3dcd_path())
    cheat_type_list = sorted(cheat_type_list)

    get_keypoints_for_all_cheat(cheat_type_list)