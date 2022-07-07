from distutils import dist
import os
import json
from turtle import screensize
import numpy as np
import pandas as pd
from scripts.loader import load_dict_from_json

#class Interiors():
#    def __init__(self) -> None:

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def distance_to_target(trial):
    # trial is the json file of a single trial
    x_target = (trial['target_bbox'][2] - trial['target_bbox'][0]) / 2 + trial['target_bbox'][0]
    y_target = (trial['target_bbox'][3] - trial['target_bbox'][1]) / 2 + trial['target_bbox'][1]
    try:
        #return float(np.sqrt((x_target - trial['response_X'])**2 + (y_target - trial['response_Y'])**2 ))
        return np.linalg.norm(np.array([x_target, y_target]) - np.array([trial['response_X'], trial['response_Y']]))
    except TypeError:
        print(trial['X'])
        print(trial['response_X'])
        print(type(trial['response_X']))
        print(x_target)
        print(type(x_target))

def target_found_response(target_box, response, response_size):
    x1, y1, x2, y2 = target_box
    return None
        
def add_responses(scanpaths_path, responses_path, calculate_features=True):
    responses = pd.read_csv(os.path.join(responses_path, 'responses_data.csv')).set_index(['subj_id','image'])
    for file in os.listdir(scanpaths_path):
        if file.endswith(".json"):
            scanpaths_file = os.path.join(scanpaths_path, file)
            subj_id = int(file.split('_')[-2][-2:])
            print('subj_id:', subj_id)
            print(scanpaths_file)
            subject_scanpaths = load_dict_from_json(scanpaths_file)
            for img, val in subject_scanpaths.items():
                val['subject_name']    = responses.loc[subj_id, img]['subj']
                # necesito corregir por el tamaño de la pantalla
                # DUDA: el target bbox esta en tamaño pantalla?
                screen_height, screen_width = float(val['screen_height']), float(val['screen_width'])
                image_height, image_width   = float(val['image_height']), float(val['image_width'])
                offset_height, offset_width = (screen_height - image_height)/2, (screen_width-image_width)/2
                val['response_X']           = float(responses.loc[subj_id, img]['response_x']) - offset_width
                val['response_Y']           = float(responses.loc[subj_id, img]['response_y']) - offset_height
                # previo a la correccion por offset
                #val['response_X']      = float(responses.loc[subj_id, img]['response_x'])
                #val['response_Y']      = float(responses.loc[subj_id, img]['response_y'])
                val['response_size']   = responses.loc[subj_id, img]['response_size']
                val['response_click']  = responses.loc[subj_id, img]['response_time_click']
                val['response_circle'] = responses.loc[subj_id, img]['response_time_circle']
                if calculate_features:
                    val['distance_to_target']  = distance_to_target(val)
                    val['target_found_response'] = bool(val['distance_to_target']  <= val['response_size']) #target_found_response(val)
                    #val['confidence_score']    = 0
                    val['delta_time_response'] = val['response_circle'] - val['response_click']
                # update dict - no estoy seguro que sea necesario
                
                subject_scanpaths[img].update(val)
                #print(subject_scanpaths[img])
                #break
            with open(os.path.join(responses_path, 'human_scanpaths', file), "w") as outfile:
                json.dump(subject_scanpaths, outfile, cls=NpEncoder)

def get_responses_features(subjs):
    df = []
    for subj, imgs in subjs.items():
        for img, data in imgs.items():
            upd = {'subj': subj, 
                    'img': img,
                    'max_fixations': data['max_fixations'],
                    'target_found': data['target_found'],
                    'target_found_response': data['target_found_response'],
                    'response_size': data['response_size'],
                    'distance_to_target': data['distance_to_target'],
                    'delta_time_response': data['delta_time_response'],
                    'response_x': data['response_X'],
                    'response_y': data['response_Y'], 
                    }
            df.append(upd)
    return pd.DataFrame(df)
    