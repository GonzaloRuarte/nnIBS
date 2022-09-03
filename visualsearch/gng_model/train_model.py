import numpy as np
from os import path
import argparse

import loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model go-no-go')
    parser.add_argument('-datapath', help="Path to the dataset", default='./../../Datasets/GNGposteriors/')
    args = parser.parse_args()
    
    # load data target present trials
    try:
        tp_data = np.load(path.join(args.datapath, 'target_present_data.npz'))
    except Exception as ex:
        print(ex)
        tp_data = np.load(path.abspath("target_present_data.npz"))
    tp_posteriors = tp_data["posteriors"]
    tp_fixation_nums = tp_data["fixations"]
    tp_labels = tp_data["labels"]
        
    tp_posteriors = np.expand_dims(tp_posteriors, axis=1) #para incorporar el canal (que es uno solo en este caso)

    # load data target absent trials
    try:
        ta_data = np.load(path.join(args.datapath, 'target_absent_data.npz'))
    except Exception as ex:
        print(ex)
        ta_data = np.load(path.abspath('target_absent_data.npz'))
    ta_posteriors = ta_data["posteriors"]
    ta_fixation_nums = ta_data["fixations"]
    ta_labels = ta_data["labels"]
        
    ta_posteriors = np.expand_dims(ta_posteriors, axis=1) #para incorporar el canal (que es uno solo en este caso)

    
    posteriors = np.concatenate((tp_posteriors,ta_posteriors),axis=0)
    labels = np.concatenate((tp_labels,ta_labels),axis=0)
    fixation_nums = np.concatenate((tp_fixation_nums,ta_fixation_nums),axis=0)
    del ta_posteriors,ta_labels,tp_posteriors,tp_labels,tp_fixation_nums,ta_fixation_nums

    model_loader = loader.ModelLoader()
    model_loader.cross_val(posteriors,labels,fixation_nums)

