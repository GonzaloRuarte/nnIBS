from os import path, mkdir, listdir, remove
import shutil
import utils
import numpy as np

""" This script requires that the COCOSearch18 images are in the folder ../images.
    Since the same image can be used for several tasks, those images are renamed as separate files for each task.
    Images are also extracted from the category folder and placed in ../images.
    Only 80% of the dataset is available, so there are some images that are never used in the human trials files.
"""

human_scanpaths_test_file = 'fixations_TA_trainval.json'

images_dir      = '../images_TA/'
targets_dir     = '../targets/'
scanpaths_dir   = '../human_scanpaths/'
categories_path = 'categories/'

image_height  = 1050
image_width   = 1680
screen_height = 1050
screen_width  = 1650

# Estimated value from IRL's model patch size
receptive_size = (52, 52)

max_fixations = 45  

if not path.exists(scanpaths_dir):
    mkdir(scanpaths_dir)

human_scanpaths = utils.load_dict_from_json(human_scanpaths_test_file)

images_tasks  = {}
unused_images = 0

initial_fixations_x = []
initial_fixations_y = []
# Computed from previous iterations
initial_fixation = (509, 816)

number_of_trials    = 0
targets_found       = 0
wrong_targets_found = 0
largest_scanpath    = 0
cropped_scanpaths   = 0
collapsed_scanpaths = 0
collapsed_fixations = 0
trivial_scanpaths   = 0

trials_properties = []
trials_processed  = []

subjects = {}
for scanpath in human_scanpaths:
    current_subject = scanpath['subject']

    if current_subject in subjects:
        subject_scanpaths = subjects[current_subject]
    else:
        subject_scanpaths = {}
        subjects[current_subject] = subject_scanpaths

    image_name = scanpath['name']
    task       = scanpath['task']
    
    # Check if the task of the trial is different this time for this image
    if not image_name in images_tasks:
        images_tasks[image_name] = {'task' : task, 'new_name' : None}
        shutil.move(images_dir + task + '/' + image_name, images_dir + image_name)
    else:
        image_info = images_tasks[image_name]
        while task != image_info['task']:
            # Remove file from subfolder
            image_path = images_dir + task + '/' + image_name
            if path.exists(image_path):
               remove(image_path)
            # Iterate through dict to define a new name for the file
            new_name = image_info['new_name']
            if new_name is None:
                new_name = str(int(image_name[0]) + 1) + image_name[1:]
                image_info['new_name'] = new_name
                images_tasks[new_name] = {'task' : task, 'new_name' : None}
                shutil.copyfile(images_dir + image_name, images_dir + new_name)

            image_info = images_tasks[new_name]
            image_name = new_name

    number_of_trials += 1
    scanpath_x  = scanpath['X']
    scanpath_y  = scanpath['Y']


    original_scanpath_length = len(scanpath_x)
    # Collapse consecutive fixations which are closer than receptive_size / 2
    scanpath_x, scanpath_y = utils.collapse_fixations(scanpath_x, scanpath_y, receptive_size)
    if len(scanpath_x) < original_scanpath_length:
        collapsed_scanpaths += 1
        collapsed_fixations += original_scanpath_length - len(scanpath_x)

    initial_fixations_x.append(scanpath_x[0])
    initial_fixations_y.append(scanpath_y[0])

    original_scanpath_len = len(scanpath_x)
    
    # Sanity check


    if current_subject < 10:
        current_subject_string = '0' + str(current_subject)
    else:
        current_subject_string = str(current_subject)

    if not image_name in trials_processed:
        # Save trial info
        target_name = image_name[:-4] + '_target' + image_name[-4:]

        trials_properties.append({'image' : image_name,  'dataset' : 'COCOSearch18 Dataset', \
            'image_height' : image_height, 'image_width' : image_width, 'initial_fixation_row' : initial_fixation[0], 'initial_fixation_column' : initial_fixation[1], \
                    'target_object' : task,'memory_set' : [target_name]})

        if not path.exists(targets_dir):
            mkdir(targets_dir)

        # Copy target
        shutil.copyfile(path.join(categories_path, task + '.jpg'), path.join(targets_dir, target_name))

        trials_processed.append(image_name)

    subject_scanpaths[image_name] = {'subject' : current_subject_string, 'dataset' : 'COCOSearch18 Dataset', 'image_height' : image_height, 'image_width' : image_width, 'memory_set': [target_name], \
        'screen_height' : screen_height, 'screen_width' : screen_width, 'receptive_height' : receptive_size[0], 'receptive_width' : receptive_size[1],  \
             'X' : scanpath_x, 'Y' : scanpath_y, 'T' : scanpath['T'], 'target_object' : scanpath['task'], 'max_fixations' : max_fixations}

# Save trials properties
utils.save_to_json('../trials_properties_TA_trainval.json', trials_properties)

# Save a human_scanpaths file for each subject
for subject in subjects:
    if subject < 10:
        subject_string = '0' + str(subject)
    else:
        subject_string = str(subject)

    subject_scanpaths_file   = 'subj' + subject_string + '_scanpaths.json'
    utils.save_to_json(path.join(scanpaths_dir, subject_scanpaths_file), subjects[subject])

# Clean up unused images
categories = [filename for filename in listdir(images_dir) if path.isdir(images_dir + filename)]
for category in categories:
    unused_images += len(listdir(images_dir + category))
    shutil.rmtree(images_dir + category)

initial_fixation_mean = (round(np.mean(initial_fixations_y)), round(np.mean(initial_fixations_x)))

print('Total targets found: ' + str(targets_found) + '/' + str(number_of_trials) + '. Wrong targets found: ' + str(wrong_targets_found))
print('Initial fixation mean: ' + str(initial_fixation_mean))
print('Number of unused images: ' + str(unused_images))
print('Largest target found scanpath: ' + str(largest_scanpath))
print("Collapsed scanpaths (discretized in size " + str(receptive_size) + ") : " + str(collapsed_scanpaths))
print("Number of fixations collapsed: " + str(collapsed_fixations))

print('Trivial scanpaths (length one): ' + str(trivial_scanpaths))
