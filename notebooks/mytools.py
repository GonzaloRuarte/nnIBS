import os
import json
import glob
from turtle import screensize
import numpy as np
import pandas as pd
import cv2
from itertools import chain
import matplotlib.pyplot as plt

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

def distance_to_last_fix(trial):
    x_target = trial['X'][-1]
    y_target = trial['Y'][-1]
    return np.linalg.norm(np.array([x_target, y_target]) - np.array([trial['response_x'], trial['response_y']]))

def distance_to_target(trial):
    # trial is the json file of a single trial
    y_target = (trial['target_bbox'][2] - trial['target_bbox'][0]) / 2 + trial['target_bbox'][0]
    x_target = (trial['target_bbox'][3] - trial['target_bbox'][1]) / 2 + trial['target_bbox'][1]
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
    y1, x1, y2, x2  = trial['target_bbox']
    side_target_x, side_target_y = x2 - x1, y2 - y1
    assert (side_target_x == side_target_y) & (side_target_y>0) & (side_target_x>0), 'Target box is not a square'
    return bool(trial['distance_to_target'] <= (side_target_x/2 + trial['response_size']))
    
def dimensions_check_scanpaths_dict(scanpaths_path, img_size_height=768, img_size_width=1024):
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
        print('Dimesions are OK, X: columns/widht, Y: rows/heigth')
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
                assert responses.loc[subj_id, img]['response_x'] <= screen_height, f'Se fue del ancho en el dato {subj_id},{img}'
                assert responses.loc[subj_id, img]['response_y'] <= screen_width, f'Se fue del ancho en el dato {subj_id},{img},'
                # estan cambiados en matlab, los cruzo
                val['response_y'] = float(responses.loc[subj_id, img]['response_x']) - offset_width
                val['response_x'] = float(responses.loc[subj_id, img]['response_y']) - offset_height
                val['response_size']   = responses.loc[subj_id, img]['response_size']
                val['response_click']  = responses.loc[subj_id, img]['response_time_click']
                val['response_circle'] = responses.loc[subj_id, img]['response_time_circle']
                # corrijo los que tenian 3 sacadas máximas y los ponemos dentro de los que tenian 4 (subs 37 y 29)
                if val['max_fixations'] == 4: val['max_fixations'] = 5
                if calculate_features:
                    val['distance_to_target']    = distance_to_target(val)
                    val['distance_to_last_fix']  = distance_to_last_fix(val)
                    val['last_fix_dur']          = -1
                    val['target_found_response'] = target_found_response(val)
                    val['delta_time_response']   = val['response_circle'] - val['response_click']
                    #val['surface_covered']     = 0
                    
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
                    'distance_to_last_fix': data['distance_to_last_fix'],
                    'delta_time_response': data['delta_time_response'],
                    'response_x': data['response_x'],
                    'response_y': data['response_y'], 
                    'target_bbox_x': data['target_bbox'][1],
                    'target_bbox_y': data['target_bbox'][0],
                    'response_target_relative_x': data['target_bbox'][1] - data['response_x'],
                    'response_target_relative_y': data['target_bbox'][0] - data['response_y']
                    }
            df.append(upd)
    return pd.DataFrame(df)

def create_scanpaths_df(dict_data, use_response=False):
    scanpath_df = []
    for subj in dict_data.keys():
        for img_f in dict_data[subj].keys():
            f = dict_data[subj][img_f]['target_found']
            scanpath_df.append([(subj,img_f,idx,x,y,t,f) for idx, (x,y,t) in enumerate(zip(dict_data[subj][img_f]['X'], 
                                                                                        dict_data[subj][img_f]['Y'],
                                                                                        dict_data[subj][img_f]['T']))])
            if use_response:
                # agregar a la ultima lista la respuesta marcada como r
                scanpath_df[-1].append((subj,img_f,'r', dict_data[subj][img_f]['response_x'],
                                                        dict_data[subj][img_f]['response_y'],
                                                        dict_data[subj][img_f]['response_size'],f))
            
    scanpaths_df = pd.DataFrame(list(chain(*scanpath_df)), columns=('subj', 'img', 'fix_order', 'x', 'y', 't','target_found'))
    return scanpaths_df

##############
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
    rx, ry = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
    r = subjs_response[subj][image_name]['response_size']
    scanpath_x = np.array(subjs_response[subj][image_name]['X'])
    scanpath_y = np.array(subjs_response[subj][image_name]['Y'])

    img = cv2.imread(os.path.join(data_path, 'images',image_name))
    tmp_files = glob.glob(os.path.join(data_path, 'templates', image_name[:-4] + '*'))
    tmp = cv2.imread(tmp_files[0])
    
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
    img = cv2.imread(os.path.join(data_path, 'images',image_name))
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
                rx, ry = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                r = subjs_response[subj][image_name]['response_size']
                if y_correction:
                    ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                else:
                    ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                n_subjs+=1
            elif use=='target_found':
                if target_f:
                    rx, ry = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                    r = subjs_response[subj][image_name]['response_size']
                    if y_correction:
                        ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    else:
                        ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    n_subjs+=1
            elif use=='target_not_found':
                if not target_f:
                    rx, ry = subjs_response[subj][image_name]['response_x'], subjs_response[subj][image_name]['response_y']
                    r = subjs_response[subj][image_name]['response_size']
                    if y_correction:
                        ax.add_patch(Circle((rx,768-ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    else:
                        ax.add_patch(Circle((rx,ry), r, fill=True, edgecolor='blue', linewidth=3, alpha=0.5))
                    n_subjs+=1
            else:
                raise ValueError('use must be one of "all", "target_found", "target_not_found"')
            
    ax.text(20,30, f'N. subjs: {n_subjs}, considerados: {use}', style='normal', fontsize=14, 
                bbox={'facecolor': 'red', 'alpha': 0.7, 'pad': 10});
    return fig, ax

    
def plot_responses_vs_target_all(responses_df):
    pass

#####
# BORRAR Y REEMPLAZAR POR GAZEHEATPOINTS
# plot density map
def gaussian_mask(sizex,sizey, sigma=33, center=None,fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x,y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey,sizex))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def fixpos2densemap(fix_arr, width, height, imgfile, alpha=0.5, threshold=10):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap 
    """
    
    heatmap = np.zeros((height,width), np.float32)
    for n_subject in range(fix_arr.shape[0]):
        heatmap += gaussian_mask(width, height, 33, (fix_arr[n_subject,0],fix_arr[n_subject,1]),
                                fix_arr[n_subject,2])

    # Normalization
    heatmap = heatmap/np.amax(heatmap)
    heatmap = heatmap*255
    heatmap = heatmap.astype("uint8")
    
    if imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask
        mask = np.where(heatmap<=threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile*mask + heatmap_color*(1-mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1-alpha, marge,alpha,0)
        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

def plot_fixposmap(data, image_file=None, image_path = None, plot_save_path=None, name=''):
    """_summary_

    Args:
        image_file (str): image file name to plot
        data (dict): the data loaded from the json file into a dictionary
        plot_save_path (_type_, optional): _description_. Defaults to None.
        image_path (str): image path
    """
    fix_list = []
    subj_num = 0
    if image_file is not None:
        if image_path is None:
            image_path = os.path.join(os.getcwd(), 'images')
        print('entro')
        # Load image file
        img = cv2.imread(os.path.join(image_path,image_file))
        H, W, _ = img.shape
        for subj in data.keys():
            if image_file in data[subj].keys():
                fix_list.append([(subj,idx,x,y,1) for idx, (x,y,t) in enumerate(zip(data[subj][image_file]['X'], 
                                                                                    data[subj][image_file]['Y'],
                                                                                    data[subj][image_file]['T']))])
                subj_num=subj
    else:
        img = np.zeros([768,1024,3],dtype=np.uint8)
        img.fill(255)
        H, W, _ = (768,1024,1)
        #image_file= 'grayscale_9_other.jpg'
        for subj in data.keys():
            for image_f in data[subj].keys():
                fix_list.append([(subj,idx,x,y,1) for idx, (x,y,t) in enumerate(zip(data[subj][image_f]['X'], 
                                                                                    data[subj][image_f]['Y'],
                                                                                    data[subj][image_f]['T']))])
    # Create heatmap
    fix_arr = pd.DataFrame(chain.from_iterable(fix_list)).iloc[:,2:5]
    heatmap = fixpos2densemap(fix_arr.to_numpy(), W, H, img, 0.5, 5)
    plt.imshow(heatmap)
    if image_file is None:
        print('fue none')
        if plot_save_path is not None:
            cv2.imwrite(os.path.join(plot_save_path, f'fixs_{name}.jpg'), heatmap)
        else:
            pass
    else: 
        print('no fue none')
        # plt.plot(data[subj_num][image_file]['initial_fixation_column'],
        #     data[subj_num][image_file]['initial_fixation_row'],'purple',
        #     marker='+',markersize=20, markeredgewidth=5);
        if plot_save_path is not None:
            cv2.imwrite(os.path.join(plot_save_path, image_file), heatmap)
    return heatmap
