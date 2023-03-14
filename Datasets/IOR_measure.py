from os import path, listdir
import json
import numpy as np

for dataset in listdir("."):
    if path.isfile(dataset) or dataset.endswith("Responses"):
        continue
    if dataset == "COCOSearch18":
        dataset = path.join(dataset,"tp_trainval")
    scanpaths_folder = path.join(dataset,"human_scanpaths")
    iors = {}

    for participant in listdir(scanpaths_folder):
        iors[participant[:-5]]= {}                
        with open(path.join(scanpaths_folder,participant), 'r') as json_file:
            human_scanpaths =  json.load(json_file)
        images_without_repetition = 0
        for image in human_scanpaths:
            
            
            x = human_scanpaths[image]["X"]
            y = human_scanpaths[image]["Y"]
            coords = np.array(list(zip(x,y)),dtype=int) // 32 #32 is the cell cize
            max_ior_image_subject = 0
            min_ior_image_subject = 0
            amount_repetitions = 0
            avg_ior_image_subject = 0
            for index in range(0,len(coords)-1):
                for repetition in range(index+1,len(coords)):
                    if (coords[index] == coords[repetition]).all():
                        if repetition - index > max_ior_image_subject:
                            max_ior_image_subject = repetition - index
                        if repetition - index < min_ior_image_subject or min_ior_image_subject == 0:
                            min_ior_image_subject = repetition - index
                        amount_repetitions += 1
                        avg_ior_image_subject += (repetition - index)    
            avg_ior_image_subject = amount_repetitions and avg_ior_image_subject / amount_repetitions or 0 
            if amount_repetitions == 0:
                images_without_repetition +=1
            else:
                iors[participant[:-5]][image] ={"max_fixations_until_return" : max_ior_image_subject,"min_fixations_until_return" : min_ior_image_subject,"avg_fixations_until_return" :avg_ior_image_subject,"amount_iors":amount_repetitions}
            iors[participant[:-5]]["images_without_repetition"] = images_without_repetition
    with open(path.abspath("iors_"+dataset.split("\\")[0]+".json"), 'w') as json_file:
        json.dump(iors,json_file, indent=4)