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
    posteriors = None
    fixation_numbers = np.array([],dtype=np.int32)
    labels = np.array([],dtype=np.int32)
    if target_present == 1:
        dataset = "tp_trainval"
    else:
        dataset = "ta_trainval"
    image_ids = np.array([],dtype=np.int32)
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
            image_fixations_dir = path.join(probability_maps_folder,image_id)
            posterior_files = listdir(image_fixations_dir)
            posterior_files.sort(key=natural_keys)
            for posterior_idx in range(0,len(posterior_files)):
                posterior = posterior_files[posterior_idx]
                posterior_file = path.join(image_fixations_dir,posterior)
                if posterior_idx == len(posterior_files) -1:
                    if target_present ==1:
                        continue
                    else:
                        label = 1
                else:
                    label = 0
                fixation_numbers = np.append(fixation_numbers,int(posterior[:-4].split('_')[1]))
                labels = np.append(labels,int(label))
                image_ids = np.append(image_ids,int(image_id))
                if posteriors is None:
                    posteriors = np.array([np.genfromtxt(posterior_file, delimiter=",")[1:25,:]])
                else:
                    posteriors = np.concatenate((posteriors,np.array([np.genfromtxt(posterior_file, delimiter=",")[1:25,:]])),axis=0)
    if target_present == 1:
        print("Target present trials")
    else:
        print("Target absent trials")
    print(f"Discarded trials:{trials_discarded}")
    print(f"Correct trials:{total_trials}")
    return posteriors,fixation_numbers,labels, image_ids

if __name__ == "__main__":
    tp_posteriors,tp_fixations,tp_labels,tp_images = parse_fixations(path.join("..","..","Results","COCOSearch18","tp_trainval_dataset","greedy_hsp","subjects_predictions"),1)
    np.savez_compressed("target_present_data.npz",posteriors=tp_posteriors,fixations=tp_fixations,labels=tp_labels,image_ids=tp_images)



    ta_posteriors,ta_fixations,ta_labels,ta_images = parse_fixations(path.join("..","..","Results","COCOSearch18","ta_trainval_dataset","greedy_hsp","subjects_predictions"),0)
    np.savez_compressed("target_absent_data.npz",posteriors=ta_posteriors,fixations=ta_fixations,labels=ta_labels,image_ids=ta_images)

