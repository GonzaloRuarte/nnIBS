import numpy as np
from os import listdir, path
import json, re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def parse_fixations(results_dir,target_present):
    fixation_locations_X = np.array([],dtype=np.int32)
    fixation_locations_Y = np.array([],dtype=np.int32)
    if target_present == 1:
        dataset = "tp_trainval"
    else:
        dataset = "ta_trainval"

    trials_discarded = 0
    total_trials = 0
    for subject in listdir(results_dir):
        subj_id = subject[-2:]

        with open("../../Datasets/COCOSearch18/"+dataset+"/human_scanpaths/subj"+subj_id+"_scanpaths.json", 'r') as json_file:
            scanpaths = json.load(json_file)
        probability_maps_folder = path.join(results_dir,subject,'probability_maps')        
        for image_id in listdir(probability_maps_folder):
            if target_present==1 and scanpaths[image_id+".jpg"]["target_found"] == False:
                trials_discarded+=1
                continue
            else:
                total_trials+=1
            scanpaths_X = scanpaths[image_id+".jpg"]["X"]
            scanpaths_Y = scanpaths[image_id+".jpg"]["Y"]
            image_fixations_dir = path.join(probability_maps_folder,image_id)
            posterior_files = listdir(image_fixations_dir)
            posterior_files.sort(key=natural_keys)
            for posterior_idx in range(0,len(posterior_files)):
                fixation_locations_X = np.append(fixation_locations_X,scanpaths_X[posterior_idx])
                fixation_locations_Y = np.append(fixation_locations_Y,scanpaths_Y[posterior_idx])

    if target_present == 1:
        print("Target present trials")
    else:
        print("Target absent trials")
    print(f"Discarded trials:{trials_discarded}")
    print(f"Correct trials:{total_trials}")
    return fixation_locations_X, fixation_locations_Y

if __name__ == "__main__":
    tp_fixations_X,tp_fixations_Y = parse_fixations(path.join("..","..","Results","COCOSearch18","tp_trainval_dataset","greedy_hsp","subjects_predictions"),1)
    np.savez_compressed("target_present_fix_locations.npz",X=tp_fixations_X,Y=tp_fixations_Y)



    ta_fixations_X,ta_fixations_Y = parse_fixations(path.join("..","..","Results","COCOSearch18","ta_trainval_dataset","greedy_hsp","subjects_predictions"),0)
    np.savez_compressed("target_absent_fix_locations.npz",X=ta_fixations_X,Y=ta_fixations_Y)

