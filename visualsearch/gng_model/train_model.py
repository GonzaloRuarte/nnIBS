import numpy as np
from os import path
import argparse

import loader

def remove_short_scanpaths(posteriors,labels,fixation_nums,image_ids):
    #obtengo los índices de comienzo y fin de scanpath
    sequence_start = np.where(fixation_nums == 1)[0]
    sequence_end = np.append(sequence_start[1:]-1,[fixation_nums.shape[0]-1])

    #filtro los de tamaño <= 4
    long_intervals = np.where(sequence_end -sequence_start >2)[0]

    sequence_start = sequence_start[long_intervals]
    sequence_end = sequence_end[long_intervals]
    sequence_intervals = np.array([sequence_start,sequence_end]).T

    useful_indexes = [list(range(x[0],x[1]+1)) for x in sequence_intervals] 
    useful_indexes = sum(useful_indexes, [])

    return posteriors[useful_indexes],labels[useful_indexes],fixation_nums[useful_indexes],image_ids[useful_indexes]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model go-no-go')
    parser.add_argument('-datapath', help="Path to the dataset, is a symlink", default='./../../Datasets/GNGposteriors/')
    args = parser.parse_args()
    

    tp_data = np.load(path.abspath("target_present_data.npz"))
    tp_posteriors = tp_data["posteriors"]
    tp_fixation_nums = tp_data["fixations"]
    tp_labels = tp_data["labels"]
    tp_image_ids = tp_data["image_ids"]    

    ta_data = np.load(path.abspath("target_absent_data.npz"))
    ta_posteriors = ta_data["posteriors"]
    ta_fixation_nums = ta_data["fixations"]
    ta_labels = ta_data["labels"]
    ta_image_ids = ta_data["image_ids"] 

    

    
    posteriors = np.concatenate((tp_posteriors,ta_posteriors),axis=0)
    labels = np.concatenate((tp_labels,ta_labels),axis=0)
    fixation_nums = np.concatenate((tp_fixation_nums,ta_fixation_nums),axis=0)
    image_ids = np.concatenate((tp_image_ids,ta_image_ids),axis=0)

    del ta_posteriors,ta_labels,tp_posteriors,tp_labels,tp_fixation_nums,ta_fixation_nums,ta_image_ids,tp_image_ids

    dataset = loader.PosteriorDataset(posteriors,labels,fixation_nums,image_ids)
    model_loader = loader.ModelLoader(dataset=dataset)
    model_loader.cross_val()
