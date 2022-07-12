import os
import json
import glob
from turtle import screensize
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from sqlalchemy import case

from scripts.loader import load_dict_from_json, load_human_scanpaths, load_trials_properties
from matplotlib.patches import Rectangle, Circle

#class Interiors():
#    def __init__(self) -> None:

def str2list(s):
    ls = s.lstrip('[').rstrip(']').split(',')
    return [float(x) for x in ls]
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
    
def dimensions_check_scapaths(scanpaths_path, img_size_height=768, img_size_width=1024):
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
    
def dimensions_check_response():
    # las respuestas son el 0,0 abajo a la izquierda o arriba a la izquierda?
    pass

def dimensions_check_target_bbox():
    pass

def add_responses(scanpaths_path, responses_path, calculate_features=True):
    responses = pd.read_csv(os.path.join(responses_path, 'responses_data.csv')).set_index(['subj_id','image'])
    trials_data = load_trials_properties(os.path.join(responses_path, '..', 'trials_properties.json'))
    trials_data = pd.DataFrame(trials_data).set_index('image')
    for file in os.listdir(scanpaths_path):
        if file.endswith(".json"):
            scanpaths_file = os.path.join(scanpaths_path, file)
            subj_id = int(file.split('_')[-2][-2:])
            print('subj_id:', subj_id)
            print('file: ', scanpaths_file)
            subject_scanpaths = load_dict_from_json(scanpaths_file)
            for img, val in subject_scanpaths.items():
                val['subject_name']    = responses.loc[subj_id, img]['subj']
                val['initial_fixation_row'] = trials_data.loc[img, 'initial_fixation_row']
                val['initial_fixation_column'] = trials_data.loc[img, 'initial_fixation_column']
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
                    'response_x': data['response_y'],
                    'response_y': data['response_x'], 
                    'target_bbox_x': data['target_bbox'][1],
                    'target_bbox_y': data['target_bbox'][0]
                    }
            df.append(upd)
    return pd.DataFrame(df)

def get_trial_scanpath_numpy(subject, image):
    pass

# plot funcs
# TODO sacar el hardcodeo del tamaño de la pantalla en alto
# TODO pensar si sacar el resp path y agregar direcatamente el dataframe
def plot_trial_subject_response(subj, image_name, data_path, resp_path, y_correction = False,
                                show_scanpath = True, ax=None):

    #subj = 41
    #image_name = 'grayscale_100_oliva.jpg' 
    subjs_response = load_human_scanpaths(os.path.join(resp_path, 'human_scanpaths'))
    target_f = subjs_response[subj][image_name]['target_found']
    max_fix  = subjs_response[subj][image_name]['max_fixations']-1
    ty, tx = subjs_response[subj][image_name]['target_bbox'][:2]
    ry, rx = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
    r = subjs_response[subj][image_name]['response_size']
    scanpath_x = np.array(subjs_response[subj][image_name]['X'])
    scanpath_y = np.array(subjs_response[subj][image_name]['Y'])

    img = cv.imread(os.path.join(data_path, 'images',image_name))
    tmp_files = glob.glob(os.path.join(data_path, 'templates', image_name[:-4] + '*'))
    tmp = cv.imread(tmp_files[0])
    
    if ax is None:
        fig, ax = plt.subplots(1,2, figsize=(15,10), gridspec_kw={'width_ratios': [3, 1]})
    ax[0].imshow(img, cmap='gray');
    ax[0].add_patch(Rectangle((tx,ty), tmp.shape[1], tmp.shape[0], fill=False, edgecolor='red', linewidth=3))
    if y_correction:
        ax[0].add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
    else:
        ax[0].add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))    
    if show_scanpath:
        for n, (x, y) in enumerate(zip(scanpath_x, scanpath_y)):
            ax[0].plot(x, y, 'co',alpha=0.9, markersize=10)
            if n != 0:
                ax[0].plot([x_prev, x], [y_prev, y], 'r--', alpha=0.9, linewidth=2)
            x_prev, y_prev = x, y
        
        ax[0].text(835,745, f'Target found: {target_f}', style='normal', fontsize=10, 
                    bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 10})
        ax[1].text(4, 8, f'Saccadic threshold: {max_fix}', style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 10})
        
    #ax[0].plot(x_init, y_init, 'g')
    ax[1].imshow(tmp, cmap='gray');
    
    return fig, ax

def plot_image_responses(image_name, data_path, resp_path, y_correction = False, use='all',ax=None):
    
    # plot image and overlay target box
    img = cv.imread(os.path.join(data_path, 'images',image_name))
    subjs_response = load_human_scanpaths(os.path.join(resp_path, 'human_scanpaths'))
    ty, tx = subjs_response[list(subjs_response.keys())[0]][image_name]['target_bbox'][:2]
    
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(15,10))
    ax.imshow(img,cmap='gray');
    ax.add_patch(Rectangle((tx,ty), 72, 72, fill=False, edgecolor='red', linewidth=3))
    
    # plot all responses circles
    n_subjs = 0
    for subj in subjs_response.keys():
        if image_name in subjs_response[subj].keys():
            target_f = subjs_response[subj][image_name]['target_found']
            if use=='all':   
                ry, rx = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                r = subjs_response[subj][image_name]['response_size']
                if y_correction:
                    ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                else:
                    ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                n_subjs+=1
            elif use=='target_found':
                if target_f:
                    ry, rx = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                    r = subjs_response[subj][image_name]['response_size']
                    if y_correction:
                        ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    else:
                        ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    n_subjs+=1
            elif use=='target_not_found':
                if not target_f:
                    ry, rx = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                    r = subjs_response[subj][image_name]['response_size']
                    if y_correction:
                        ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    else:
                        ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    n_subjs+=1
            else:
                raise ValueError('use must be one of "all", "target_found", "target_not_found"')
            
    ax.text(20,30, f'N. subjs: {n_subjs}, considerados: {use}', style='normal', fontsize=14, 
                bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 10})
    return fig, ax

    
def plot_responses_vs_target_all(responses_df):
    pass