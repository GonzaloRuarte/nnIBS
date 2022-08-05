import run_visualsearch
from os import listdir, path
from visualsearch.utils import utils
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import numba
human_scanpaths_path = 'data/human_scanpaths'

def main():
    for subject in listdir(human_scanpaths_path):
        subject_number  = subject[4:6]
        config_filename = 'ssim'
        run_visualsearch.main(config_filename, None, None, int(subject_number), 'all', True)
        output_path = path.join(path.join('output', + config_filename), 'human_subject_' + subject_number)
        human_scanpaths = utils.load_from_json(path.join(output_path, 'Subject_scanpaths.json'))
        for image_name in human_scanpaths:
            scanpath = human_scanpaths[image_name]
            human_fixations_x = np.array(scanpath['X'], dtype=np.int)
            human_fixations_y = np.array(scanpath['Y'], dtype=np.int)
            probability_maps_folder = path.join(output_path, path.join('probability_maps', image_name[:-4]))
            for index in range(1, np.size(human_fixations_x)): #la fijacion inicial es compartida, por ende no hay que compararla                
                probability_map = pd.read_csv(probability_maps_folder + 'fixation_' + str(index))                
                compute_scanpath_prediction_metrics(probability_map, human_fixations_y[:index], human_fixations_x[:index])

def center_gaussian(shape):
    sigma  = [[1, 0], [0, 1]]
    mean   = [shape[0] // 2, shape[1] // 2]
    x_range = np.linspace(0, shape[0], shape[0])
    y_range = np.linspace(0, shape[1], shape[1])

    x_matrix, y_matrix = np.meshgrid(y_range, x_range)
    quantiles = np.transpose([y_matrix.flatten(), x_matrix.flatten()])
    mvn = multivariate_normal.pdf(quantiles, mean=mean, cov=sigma)
    mvn = np.reshape(mvn_at_fixation, shape)

    return mvn

def compute_scanpath_prediction_metrics(probability_map, human_fixations_y, human_fixations_x):
    probability_map = probability_map.to_numpy(dtype=np.float)
    probability_map = normalize(probability_map)
    baseline_map    = center_gaussian(probability_map.shape)

    roc = np.mean(AUCs(probability_map, human_fixations_y, human_fixations_x)) # ¿Promediamos?
    nss = NSS(probability_map, human_fixations_y, human_fixations_x)
    ig  = infogain(probability_map, baseline_map, human_fixations_y, human_fixations_x)

    #en que formato guardar los resultados? json?

def normalize(probability_map):
    normalized_probability_map = probability_map - np.min(probability_map)
    normalized_probability_map = normalized_probability_map / np.max(normalized_probability_map)

    return normalized_probability_map

def NSS(saliency_map, ground_truth_fixations_y, ground_truth_fixations_x):
    mean = np.mean(saliency_map)
    std = np.std(saliency_map)
    value = np.copy(saliency_map[ground_truth_fixations_y, ground_truth_fixations_x])
    value -= mean

    if std:
        value /= std

    return value

def infogain(s_map, baseline_map, ground_truth_fixations_y, ground_truth_fixations_x):
	eps = 2.2204e-16

	s_map        = s_map / (np.sum(s_map) * 1.0)
	baseline_map = baseline_map / (np.sum(baseline_map) * 1.0)

	temp = []

	for i in zip(ground_truth_fixations_y, ground_truth_fixations_x):
		temp.append(np.log2(eps + s_map[i[1], i[0]]) - np.log2(eps + baseline_map[i[1], i[0]]))

	return np.mean(temp)

def AUCs(probability_map, ground_truth_fixations_y, ground_truth_fixations_x):
    """ Calculate AUC scores for fixations """
    rocs_per_fixation = []

    for i in tqdm(range(len(ground_truth_fixations_x)), total=len(ground_truth_fixations_x)):
        positive  = probability_map[ground_truth_fixations_y[i], ground_truth_fixations_x[i]]
        negatives = probability_map.flatten()

        this_roc = auc_for_one_positive(positive, negatives)
        rocs_per_fixation.append(this_roc)

    return np.asarray(rocs_per_fixation)

@numba.jit(nopython=True)
def fill_fixation_map(fixation_map, fixations):
    """fixationmap: 2d array. fixations: Nx2 array of y, x positions"""
    for i in range(len(fixations)):
        fixation_y, fixation_x = fixations[i]
        fixation_map[int(fixation_y), int(fixation_x)] += 1


def auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    return _auc_for_one_positive(positive, np.asarray(negatives))


@numba.jit(nopython=True)
def _auc_for_one_positive(positive, negatives):
    """ Computes the AUC score of one single positive sample agains many negatives.
    The result is equal to general_roc([positive], negatives)[0], but computes much
    faster because one can save sorting the negatives.
    """
    count = 0
    for negative in negatives:
        if negative < positive:
            count += 1
        elif negative == positive:
            count += 0.5

    return count / len(negatives)

if __name__ == "__main__":
    main()