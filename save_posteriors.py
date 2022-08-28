

from os import path, listdir
import re

import main as nnIBS


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def get_subjects_posteriors_for_model(model_name,dataset_name):

    

    human_scanpaths_files = sorted_alphanumeric(listdir(path.join(path.join("Datasets", dataset_name),"human_scanpaths")))



    for subject in human_scanpaths_files:
        subject_number = subject[4:6]

        print('[Human Scanpath Prediction] Running ' + model_name + ' on ' + dataset_name + ' dataset using subject ' + subject_number + ' scanpaths')
        nnIBS.main(dataset_name, int(subject_number))
    



get_subjects_posteriors_for_model("nnIBS","COCOSearch18/ta_trainval")