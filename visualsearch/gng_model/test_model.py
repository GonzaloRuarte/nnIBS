import numpy as np
from os import path
import argparse


import loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model go-no-go')
    parser.add_argument('-datapath', help="Path to the dataset, is a symlink", default='./../../Datasets/GNGposteriors/')
    args = parser.parse_args()
    
    tp_data = np.load(path.abspath("/home/liaa-user/repos/posteriors-cocosearch18-subjects/target_present_data_no_oracle.npz"))
    tp_posteriors = tp_data["posteriors"]
    tp_fixation_nums = tp_data["fixations"]
    tp_labels = tp_data["labels"]
    
    posteriors = np.concatenate((tp_posteriors,tp_posteriors),axis=0)
    labels = np.concatenate((tp_labels,tp_labels),axis=0)
    fixation_nums = np.concatenate((tp_fixation_nums,tp_fixation_nums),axis=0)
    del tp_posteriors,tp_labels,tp_fixation_nums

    model_loader = loader.ModelLoader()
    model_loader.predict(posteriors,labels,fixation_nums)

