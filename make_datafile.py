import json
import numpy as np
import pickle
import csv
import os
from os import listdir
from os.path import isfile, join


dataset = ['train','test','val']

how2sign_root = './how2sign.github.io/How2Sign/sentence_level'
json_postfix = 'rgb_front/features/openpose_output/json'
text_postfix = 'text/en/raw_text/re_aligned/how2sign_realigned_'

for target in dataset:
    csv_path = os.path.join( os.path.join(how2sign_root,target), text_postfix+target+'.csv')
    print(csv_path)
    json_dir =   os.path.join(os.path.join(how2sign_root, target), json_postfix)
    print(json_dir)

    ttl_data = []
    cnt = 0
    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(csv_reader)
        for row in csv_reader:

            data = {}
            sentence_name = row[3]
            sentence = row[-1]
            data['sentence'] = sentence

            kpts_list = []

            dir_name = os.path.join(json_dir, sentence_name)
            
            if os.path.exists(dir_name):
                for f in sorted(listdir(dir_name)):
                    if os.path.exists(os.path.join(dir_name,f)) and  isfile(os.path.join(dir_name, f)):
                        with open(os.path.join(dir_name, f), "r") as slp_json:
                            try:
                                slp_kpts = json.load(slp_json)
                                pose_kpts = np.reshape(slp_kpts['people'][0]['pose_keypoints_2d'],(-1,3))[:,:2]
                                face_kpts = np.reshape(slp_kpts['people'][0]['face_keypoints_2d'],(-1,3))[:,:2]
                                lhand_kpts = np.reshape(slp_kpts['people'][0]['hand_left_keypoints_2d'],(-1,3))[:,:2]
                                rhand_kpts = np.reshape(slp_kpts['people'][0]['hand_right_keypoints_2d'],(-1,3))[:,:2]

                                ttl_kpts=np.concatenate([pose_kpts, face_kpts, lhand_kpts, rhand_kpts])
                                kpts_list.append(ttl_kpts)
                            except Exception as e:
                                print(e)
                                print(f'Error while opening {os.path.join(dir_name,f)}')
                
                if cnt % 1000 ==0:
                    print(f'{cnt}.. with dir_name {dir_name}') 
                                
                data['ttl_kpts_seq'] = np.array(kpts_list)

                if len(data['ttl_kpts_seq'])>0:
                    ttl_data.append(data)
                    cnt += 1


    with open(f'{target}.pkl', 'wb') as f:
        pickle.dump(ttl_data, f, pickle.HIGHEST_PROTOCOL)