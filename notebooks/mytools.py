import os
import json
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
        return float(np.sqrt((x_target - trial['response_X'])**2 + (y_target - trial['response_Y'])**2 ))
    except TypeError:
        print(trial['X'])
        print(trial['response_X'])
        print(type(trial['response_X']))
        print(x_target)
        print(type(x_target))
        
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
                val['response_X']      = float(responses.loc[subj_id, img]['response_x'])
                val['response_Y']      = float(responses.loc[subj_id, img]['response_y'])
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
