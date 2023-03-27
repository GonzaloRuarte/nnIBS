from os import path, listdir
import json
import numpy as np

for dataset in listdir("Interiors_dataset/default"):

    scanpaths_file = path.join("Interiors_dataset/default",dataset,"Scanpaths.json")
    iors = {}
            
    with open(scanpaths_file, 'r') as json_file:
        scanpaths =  json.load(json_file)
    images_without_repetition = 0
    for image in scanpaths:
        
        
        x = scanpaths[image]["X"]
        y = scanpaths[image]["Y"]
        coords = np.array(list(zip(x,y))) 
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
            iors[image] ={"max_fixations_until_return" : max_ior_image_subject,"min_fixations_until_return" : min_ior_image_subject,"avg_fixations_until_return" :avg_ior_image_subject,"amount_returns":amount_repetitions}
        iors["images_without_repetition"] = images_without_repetition
    with open(path.abspath("returns_"+dataset.split("\\")[0]+".json"), 'w') as json_file:
        json.dump(iors,json_file, indent=4)