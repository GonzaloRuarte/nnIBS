from os import path, listdir
import json
import numpy as np

def are_within_boundaries(top_left_coordinates, bottom_right_coordinates, top_left_coordinates_to_compare, bottom_right_coordinates_to_compare):
    return top_left_coordinates[0] >= top_left_coordinates_to_compare[0] and top_left_coordinates[1] >= top_left_coordinates_to_compare[1] \
         and bottom_right_coordinates[0] < bottom_right_coordinates_to_compare[0] and bottom_right_coordinates[1] < bottom_right_coordinates_to_compare[1]

for dataset in listdir("Interiors_dataset/default"):
    trials_properties_file = path.join("../Datasets/Interiors","trials_properties.json")
    scanpaths_file = path.join("Interiors_dataset/default",dataset,"Scanpaths.json")
    iors = {}
            
    with open(scanpaths_file, 'r') as json_file:
        scanpaths =  json.load(json_file)
    with open(trials_properties_file, 'r') as json_file:
        trials_properties =  json.load(json_file)
    target_bboxes = {}
    for trial in trials_properties:
        target_bboxes[trial["image"]] = [trial['target_matched_row'], trial['target_matched_column'], \
                                            trial['target_height'] + trial['target_matched_row'], trial['target_width'] + trial['target_matched_column']]
    images_without_repetition = 0
    for image in scanpaths:
        target_bbox = target_bboxes[image]
        target_bbox_in_grid = np.array(target_bbox, dtype=int) //32
        x = scanpaths[image]["X"]
        y = scanpaths[image]["Y"]
        coords = np.array(list(zip(x,y))) 
        max_ior_image_subject = 0
        min_ior_image_subject = 0
        amount_repetitions = 0
        amount_leave_target = 0
        avg_ior_image_subject = 0
        target_found = False
        for index in range(0,len(coords)-1):
            for repetition in range(index+1,len(coords)):
                if (coords[index] == coords[repetition]).all():
                    if repetition - index > max_ior_image_subject:
                        max_ior_image_subject = repetition - index
                    if repetition - index < min_ior_image_subject or min_ior_image_subject == 0:
                        min_ior_image_subject = repetition - index
                    amount_repetitions += 1

                    avg_ior_image_subject += (repetition - index)  
            if are_within_boundaries(coords[index], coords[index], (target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2] + 1, target_bbox_in_grid[3] + 1)):
                if target_found == True:
                    amount_leave_target +=1
                    target_found == False
                    print(dataset + " - " + image +" - "+ str(index) +" - "+str(coords[index])+" - "+str(target_bbox_in_grid))
                else:
                    target_found = True
                    

                 
        avg_ior_image_subject = amount_repetitions and avg_ior_image_subject / amount_repetitions or 0 
        if amount_repetitions == 0:
            images_without_repetition +=1
        else:
            iors[image] ={"amount_leave_target":amount_leave_target,"max_fixations_until_return" : max_ior_image_subject,"min_fixations_until_return" : min_ior_image_subject,"avg_fixations_until_return" :avg_ior_image_subject,"amount_returns":amount_repetitions}
        iors["images_without_repetition"] = images_without_repetition
    with open(path.abspath("returns_"+dataset.split("\\")[0]+".json"), 'w') as json_file:
        json.dump(iors,json_file, indent=4)