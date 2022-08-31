import numpy as np
from os import listdir, path




def parse_fixations(results_dir,label):
    posteriors = None
    fixation_numbers = np.array([],dtype=np.float32)
    
    for subject in listdir(results_dir):
        probability_maps_folder = path.join(results_dir,subject,'probability_maps')
        for image_id in listdir(probability_maps_folder):
            image_fixations_dir = path.join(probability_maps_folder,image_id)
            for posterior in listdir(image_fixations_dir):
                posterior_file = path.join(image_fixations_dir,posterior)
                fixation_numbers = np.append(fixation_numbers,float(posterior[:-4].split('_')[1]))
                if posteriors is None:
                    posteriors = np.array([np.genfromtxt(posterior_file, delimiter=",")[1:25,:]])
                else:
                    posteriors = np.concatenate((posteriors,np.array([np.genfromtxt(posterior_file, delimiter=",")[1:25,:]])),axis=0)

    labels = np.full(posteriors.shape[0],label,dtype=np.int32)

    return posteriors,fixation_numbers,labels

if __name__ == "__main__":
    tp_posteriors,tp_fixations,tp_labels = parse_fixations(path.join("..","..","Results","COCOSearch18","tp_trainval_dataset","greedy_hsp","subjects_predictions"),1)
    np.savez_compressed("target_present_data.npz",posteriors=tp_posteriors,fixations=tp_fixations,labels=tp_labels)



    ta_posteriors,ta_fixations,ta_labels = parse_fixations(path.join("..","..","Results","COCOSearch18","ta_trainval_dataset","greedy_hsp","subjects_predictions"),0)
    np.savez_compressed("target_absent_data.npz",posteriors=ta_posteriors,fixations=ta_fixations,labels=ta_labels)

