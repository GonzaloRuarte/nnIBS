from .models.bayesian_model import BayesianModel
from .models.greedy_model   import GreedyModel
from Metrics.scripts import human_scanpath_prediction
from .utils import utils
from . import prior
import numpy as np
import time
import importlib
from scipy.stats import entropy
from .gng_model.loader import ModelLoader
#from os import path
from .visibility_map import VisibilityMap
from .grid import Grid
import sys
from os import mkdir, path
import torch

class VisualSearcher: 
    def __init__(self, config, dataset_info, trials_properties, output_path, human_scanpaths,sigma):
        " Creates a new instance of the visual search model "
        """ Input:
                Config (dict). One entry. Fields:
                    search_model      (string)   : bayesian, greedy
                    target_similarity (string)   : correlation, geisler, ssim, ivsn
                    prior             (string)   : deepgaze, mlnet, flat, center
                    max_saccades      (int)      : maximum number of saccades allowed
                    cell_size         (int)      : size (in pixels) of the cells in the grid
                    scale_factor      (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    additive_shift    (int)      : modulates the variance of target similarity and prevents 1 / d' from diverging in bayesian search
                    save_probability_maps (bool) : indicates whether to save the probability map to a file after each saccade or not
                    proc_number       (int)      : number of processes on which to execute bayesian search
                    image_size        (int, int) : image size on which the model will operate
                    save_similarity_maps (bool)  : indicates whether to save the target similarity map for each image in bayesian search
                Dataset info (dict). One entry. Fields:
                    name          (string)         : name of the dataset
                    images_dir    (string)         : folder path where search images are stored
                    targets_dir   (string)         : folder path where the targets are stored
                    saliency_dir  (string)         : folder path where the saliency maps are stored
                    target_similarity_dir (string) : folder path where the target similarity maps are stored
                    image_height  (int)            : default image height (in pixels)
                    image_width   (int)            : default image width (in pixels)
                Trials properties (dict):
                    Each entry specifies the data of the image on which to run the visual search model. Fields:
                    image  (string)               : image name (where to look)
                    target (string)               : name of the target image (what to look for)
                    target_matched_row (int)      : starting Y coordinate, in pixels, of the target in the image
                    target_matched_column (int)   : starting X coordinate, in pixels, of the target in the image
                    target_height (int)           : height of the target in pixels
                    target_width (int)            : width of the target in pixels
                    initial_fixation_row (int)    : row of the first fixation on the image
                    initial_fixation_column (int) : column of the first fixation on the image
                Output path     (string)        : folder path where scanpaths and probability maps will be stored
                human_scanpaths (dict) : if not empty, it contains the human scanpaths which the model will use as fixations
        """
                
        self.cell_size  = config['cell_size']
        self.config = config
        self.dataset_name = dataset_info['dataset_name']
        self.model_image_size = config['image_size']
        self.max_saccades             = config['max_saccades']
        self.init_max_saccades        = config['max_saccades']
        self.grid                     = Grid(np.array(self.model_image_size), self.cell_size)
        self.scale_factor             = config['scale_factor']
        self.additive_shift           = config['additive_shift']
        self.seed                     = config['seed']
        self.save_probability_maps    = config['save_probability_maps']
        self.save_similarity_maps     = config['save_similarity_maps']
        self.number_of_processes      = config['proc_number']
        self.visibility_map           = VisibilityMap(self.model_image_size, self.grid, sigma)
        self.search_model             = self.initialize_model(config['search_model'], config['norm_cdf_tolerance'])
        self.target_similarity_dir    = dataset_info['target_similarity_dir']
        self.target_similarity_method = config['target_similarity']
        self.output_path              = output_path
        self.human_scanpaths          = human_scanpaths
        self.history_size             = config['history_size']
        self.trials_properties        = trials_properties
        self.images_dir            = dataset_info['images_dir']
        self.targets_dir           = dataset_info['targets_dir']
        self.saliency_dir          = dataset_info['saliency_dir']
        self.target_similarity_dir = dataset_info['target_similarity_dir']
        self.prior_name    = config['prior']
        self.image_size = (dataset_info['image_height'], dataset_info['image_width'])

        self.cell_size  = config['cell_size']

        # Rescale human scanpaths' coordinates (if any) to those of the grid
        utils.rescale_scanpaths(self.grid, self.human_scanpaths)




    def run(self):
        """
            Output:
                Output_path/scanpaths/Scanpaths.json: Dictionary indexed by image name where each entry contains the scanpath for that given image, alongside the configuration used.
                Output_path/probability_maps/: In this folder, the probability map computed for each saccade is stored. This is done for every image in trials_properties. (Only if save_probability_maps is true.)
                Output_path/similarity_maps/: In this folder, the target similarity map computed for each image is stored. This is done for every image in trials_properties. (Only if save_similarity_maps is true.)
        """
        
        print('Press Ctrl + C to interrupt execution and save a checkpoint \n')

        # If resuming execution, load previously generated data
        if not path.exists(self.output_path):
            mkdir(path.abspath(self.output_path))
        scanpaths, targets_found, previous_time = utils.load_data_from_checkpoint(self.output_path)

        trial_number = len(scanpaths)
        total_trials = len(self.trials_properties) + trial_number
        start = time.time()
        try:
            for trial in self.trials_properties:
                trial_number += 1
                image_name  = trial['image']
                    
                if not ('memory_set' in trial):
                    target_name = trial['target']
                    trial['memory_set'] = [target_name]
                    
                memory_set = list(map(lambda x: utils.load_image(self.targets_dir,x),trial['memory_set']))
                    
                print('Searching in image ' + image_name + ' (' + str(trial_number) + '/' + str(total_trials) + ')...')
                
                image       = utils.load_image(self.images_dir, image_name, self.model_image_size)

                image_prior = prior.load(image, image_name, self.model_image_size, self.prior_name, self.saliency_dir)
                
                initial_fixation = (trial['initial_fixation_row'], trial['initial_fixation_column'])
                initial_fixation = [utils.rescale_coordinate(initial_fixation[i], self.image_size[i], self.model_image_size[i]) for i in range(len(initial_fixation))]
                if "target_matched_row" in trial:
                    target_bbox      = [trial['target_matched_row'], trial['target_matched_column'], \
                                            trial['target_height'] + trial['target_matched_row'], trial['target_width'] + trial['target_matched_column']]
                    target_bbox      = [utils.rescale_coordinate(target_bbox[i], self.image_size[i % 2 == 1], self.model_image_size[i % 2 == 1]) for i in range(len(target_bbox))]
                else:
                    target_bbox = None
                trial_scanpath = self.search(image_name, image, image_prior, memory_set,trial['memory_set'], target_bbox, initial_fixation)

                if trial_scanpath:
                    # If there were no errors, save the scanpath
                    utils.add_scanpath_to_dict(image_name, trial_scanpath, target_bbox, trial['target_object'], self.grid, self.config, self.dataset_name, scanpaths,trial['memory_set'])
                    targets_found += trial_scanpath['target_found']
        except KeyboardInterrupt:
            time_elapsed = time.time() - start + previous_time
            utils.save_checkpoint(self.config, scanpaths, targets_found, self.trials_properties, time_elapsed, self.output_path)        
            sys.exit(0)

        time_elapsed = time.time() - start + previous_time
        if self.human_scanpaths:
            utils.save_scanpaths(self.output_path, self.human_scanpaths, filename='Subject_scanpaths.json')
        else:
            utils.save_scanpaths(self.output_path, scanpaths)
        utils.erase_checkpoint(self.output_path)

        print('Total targets found: ' + str(targets_found) + '/' + str(len(scanpaths)))
        print('Total time elapsed:  ' + str(round(time_elapsed, 4))   + ' seconds')

    
    
    def search(self, image_name, image, image_prior, memory_set, memory_set_names,target_bbox, initial_fixation):
        " Given an image, a target, and a prior of that image, it looks for the object in the image, generating a scanpath "
        """ Input:
            Specifies the data of the image on which to run the visual search model. Fields:
                image_name (string)         : name of the image
                image (2D array)            : search image
                image_prior (2D array)      : grayscale image with values between 0 and 1 that serves as prior
                memory_set (3D array)       : stimuli to search
                memory_set_names (array)    : filenames of the stimuli
                target_bbox (array)         : bounding box (upper left row, upper left column, lower right row, lower right column) of the target inside the search image
                initial_fixation (int, int) : row and column of the first fixation on the search image
            Output:
                image_scanpath   (dict)      : scanpath made by the model on the search image, alongside a 'target_found' field which indicates if the target was found
                probability_maps (csv files) : if self.save_probability_maps is True, the probability map for each saccade is stored in a .csv file inside a folder in self.output_path 
                similarity_maps  (png files) : if self.save_similarity_maps is True, the target similarity map for each image is stored inside a folder in self.output_path
        """
        # Convert prior to grid
        image_prior = self.grid.reduce(image_prior, mode='mean')
        grid_size   = self.grid.size()
        # Check prior dimensions
        if not(image_prior.shape == grid_size):
            print(image_name + ': prior image\'s dimensions don\'t match dataset\'s dimensions')
            return {}
        # Sum probabilities
        image_prior = prior.sum(image_prior, self.init_max_saccades)

        #gng_model = ModelLoader(num_classes=2)

        #gng_model.load(path.abspath("visualsearch/gng_model/GNG_model_dict.pth"))
        # Convert target bounding box to grid cells
        if target_bbox != None:
            target_bbox_in_grid = np.empty(len(target_bbox), dtype=np.int)
            target_bbox_in_grid[0], target_bbox_in_grid[1] = self.grid.map_to_cell((target_bbox[0], target_bbox[1]))
            target_bbox_in_grid[2], target_bbox_in_grid[3] = self.grid.map_to_cell((target_bbox[2], target_bbox[3]))
            if not(utils.are_within_boundaries((target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2], target_bbox_in_grid[3]), np.zeros(2), grid_size)):
                print(image_name + ': target bounding box is outside of the grid')
                return {}

        if self.human_scanpaths: 
            # Get subject scanpath for this image
            current_human_scanpath  = self.human_scanpaths[image_name]
            current_human_fixations = np.array(list(zip(current_human_scanpath['Y'], current_human_scanpath['X'])))
            self.max_saccades       = current_human_fixations.shape[0] - 1
            # Check if the probability maps have already been computed and stored
            if utils.exists_probability_maps_for_image(image_name, self.output_path):
                print('Loaded previously computed probability maps for image ' + image_name)
                human_scanpath_prediction.save_scanpath_prediction_metrics(current_human_scanpath, image_name, self.output_path)
                return {}
        
        # Initialize fixations matrix
        fixations = np.empty(shape=(self.max_saccades + 1, 2), dtype=int)
        if self.human_scanpaths:
            fixations[0] = current_human_fixations[0]
        else:
            fixations[0] = self.grid.map_to_cell(initial_fixation)
        if not(utils.are_within_boundaries(fixations[0], fixations[0], np.zeros(2), grid_size)):
            print(image_name + ': initial fixation falls off the grid')
            return {}


        target_similarity_map = np.array(list(map(lambda i: self.initialize_target_similarity_map(memory_set[i], image,target_bbox, image_name,memory_set_names[i]),range(0,len(memory_set)))))

        # Initialize variables for computing each fixation        
        likelihood = np.zeros(shape=grid_size)

        #target_selection_rate = 4 #o un random int

        #fixations_until_target_renewal = 0

        # Search
        print('Fixation:', end=' ')
        target_found = False
        start = time.time()
        searched_object_indexes = []
        if self.history_size != None:
            history_likelihoods = np.zeros(shape=(self.history_size,grid_size[0],grid_size[1]))
        for fixation_number in range(self.max_saccades + 1):
            if self.human_scanpaths:
                current_fixation = current_human_fixations[fixation_number]
            else:
                current_fixation = fixations[fixation_number]

            print(fixation_number + 1, end=' ')
            if target_bbox != None:
                if utils.are_within_boundaries(current_fixation, current_fixation, (target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2] + 1, target_bbox_in_grid[3] + 1)):
                    target_found = True
                    fixations = fixations[:fixation_number + 1]
                    break

            # If the limit has been reached, don't compute the next fixation
            if fixation_number == self.max_saccades:
                break
            
            target_similarities = np.array(list(map(lambda x: x.at_fixation(current_fixation),target_similarity_map)))
            #if fixations_until_target_renewal == 0:
                #fixations_until_target_renewal = target_selection_rate

                #selected_likelihood_index = np.random.randint(len(memory_set)) #A random target similarity map is used
            #fixations_until_target_renewal -= 1
            
            selected_likelihood_index = np.argmin(list(map(lambda x : entropy(x.flatten()),target_similarities))) #The target similarity map with minimum entropy is used
            searched_object_indexes.append(selected_likelihood_index)
            if self.history_size != None:
                history_likelihoods = np.append(history_likelihoods,[np.zeros(shape=grid_size)], axis=0)
                history_likelihoods = history_likelihoods + (target_similarities[selected_likelihood_index] * (np.square(self.visibility_map.at_fixation(current_fixation))))
                likelihood = history_likelihoods[0] #I remember the last n fixations and I also include the information gained in the current fixation
                history_likelihoods = history_likelihoods[1:] #I discard the oldest fixation info
            else:
                likelihood = likelihood + (target_similarities[selected_likelihood_index] * (np.square(self.visibility_map.at_fixation(current_fixation))))


            likelihood_times_prior = image_prior * np.exp(likelihood)
            
            marginal  = np.sum(likelihood_times_prior)
            
            posterior = likelihood_times_prior / marginal            
            
            #if not gng_model.continue_search(posterior):
                #if target_bbox != None:
                    #if utils.are_within_boundaries(current_fixation, current_fixation, (target_bbox_in_grid[0], target_bbox_in_grid[1]), (target_bbox_in_grid[2] + 1, target_bbox_in_grid[3] + 1)):
                        #target_found = True
                #fixations = fixations[:fixation_number + 1]
                #break
            fixations[fixation_number + 1] = self.search_model.next_fixation(posterior, image_name, fixation_number, self.output_path)

        end = time.time()

        if target_found:
            print('\nTarget found!')
        else:
            print('\nTarget NOT FOUND!')
        print('Time elapsed: ' + str(end - start) + '\n')

        # Revert back to pixels
        # fixations = [self.grid.map_cell_to_pixels(fixation) for fixation in fixations]

        # Note: each x coordinate refers to a column in the image, and each y coordinate refers to a row in the image
        scanpath_x_coordinates = self.get_coordinates(fixations, axis=1)
        scanpath_y_coordinates = self.get_coordinates(fixations, axis=0)

        if self.human_scanpaths:
            human_scanpath_prediction.save_scanpath_prediction_metrics(current_human_scanpath, image_name, self.output_path)

        return { 'searched_object_indexes':searched_object_indexes,'target_found' : target_found, 'scanpath_x' : scanpath_x_coordinates, 'scanpath_y' : scanpath_y_coordinates }
    
    def get_coordinates(self, fixations, axis):
        fixations_as_list = np.array(fixations).flatten()

        return [fixations_as_list[fix_number] for fix_number in range(axis, len(fixations_as_list), 2)]

    def initialize_model(self, search_model, norm_cdf_tolerance):
        if search_model == 'greedy':
            return GreedyModel(self.save_probability_maps)
        else:
            return BayesianModel(self.grid.size(), self.visibility_map, norm_cdf_tolerance, self.number_of_processes, self.save_probability_maps)

    def initialize_target_similarity_map(self,  target, image, target_bbox, image_name,stim_name):
        # Load corresponding module, which has the same name in lower case
        module = importlib.import_module('.target_similarity.' + self.target_similarity_method.lower(), 'visualsearch')
        # Get the class
        target_similarity_class = getattr(module, self.target_similarity_method.capitalize())
        target_similarity_map   = target_similarity_class(image_name, stim_name, image, target, target_bbox, self.visibility_map, self.scale_factor, self.additive_shift, self.grid, self.seed, \
            self.number_of_processes, self.save_similarity_maps, self.target_similarity_dir)
        return target_similarity_map

