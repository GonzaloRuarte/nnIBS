import os
import json
from turtle import screensize
import numpy as np
import pandas as pd
from scripts.loader import load_dict_from_json, load_human_scanpaths

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
        return np.linalg.norm(np.array([x_target, y_target]) - np.array([trial['response_x'], trial['response_y']]))
    except TypeError:
        print(trial['X'])
        print(trial['response_x'])
        print('Tipo de dato de response X: ', type(trial['response_X']))
        print(x_target)
        print(type(x_target))

def target_found_response(trial):
    x1, y1, x2, y2  = trial['target_bbox']
    side_target_x, side_target_y = x2 - x1, y2 - y1
    assert side_target_x == side_target_y; 'Target box is not a square'
    return bool(trial['distance_to_target'] <= (side_target_x/2 + trial['response_size']))
    
def dimensions_check(scanpaths_path, img_size_height=768, img_size_width=1024):
    subjs = load_human_scanpaths(scanpaths_path)
    lim_sup_x, lim_inf_x, lim_sup_y, lim_inf_y = [], [], [], []
    for subj in subjs.keys():
        for img in subjs[subj].keys():
            if np.array(subjs[subj][img]['X']).max() > img_size_width:
                lim_sup_x.append((subj, img))
            if np.array(subjs[subj][img]['X']).min() < 0:
                lim_inf_x.append((subj, img))
            if np.array(subjs[subj][img]['Y']).max() > img_size_height:
                lim_sup_y.append((subj, img))
            if np.array(subjs[subj][img]['Y']).min() < 0:
                lim_inf_y.append((subj, img))
    if any(map(lambda x: len(x) > 0, [lim_sup_x, lim_inf_x, lim_sup_y, lim_inf_y])):
        print('There are some subjects with some images with coordinates out of bounds. Potentially wrong indexes X and Y.')
        return lim_sup_x, lim_inf_x, lim_sup_y, lim_inf_y       
    else:
        print('Dimesions are OK, X: columns, Y: rows')
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
                val['response_x'] = float(responses.loc[subj_id, img]['response_x']) - offset_width
                val['response_y'] = float(responses.loc[subj_id, img]['response_y']) - offset_height
                #val['response_X'] = list(val['X']).append(val['response_x'])
                #print('Tipo de dato de X: ', type(val['X']))
                #break
                #val['response_Y'] = list(val['Y']).append(val['response_y'])
                # previo a la correccion por offset
                #val['response_X']      = float(responses.loc[subj_id, img]['response_x'])
                #val['response_Y']      = float(responses.loc[subj_id, img]['response_y'])
                val['response_size']   = responses.loc[subj_id, img]['response_size']
                val['response_click']  = responses.loc[subj_id, img]['response_time_click']
                val['response_circle'] = responses.loc[subj_id, img]['response_time_circle']
                if calculate_features:
                    val['distance_to_target']  = distance_to_target(val)
                    val['target_found_response'] = target_found_response(val)
                    #val['surface_covered']    = 0
                    val['delta_time_response'] = val['response_circle'] - val['response_click']
                
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
                    'response_x': data['response_x'],
                    'response_y': data['response_y'], 
                    }
            df.append(upd)
    return pd.DataFrame(df)

def get_trial_scanpath_numpy(subject, image):
    pass

def plot_response(subject, image, ax=None):
    
    pass
    