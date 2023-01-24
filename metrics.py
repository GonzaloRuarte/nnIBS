from Metrics.scripts import constants
import argparse
from Metrics.scripts.multimatch import Multimatch
from Metrics.scripts.human_scanpath_prediction import HumanScanpathPrediction
from Metrics.scripts.cumulative_performance import CumulativePerformance
from Metrics.scripts import utils
from os import path,listdir
import main as nnIBS

def main(datasets, compute_cumulative_performance, compute_multimatch, compute_human_scanpath_prediction):
    datasets_results = {}
    model_name = "nnIBS"
    for dataset_name in datasets:
        dataset_path = path.join(constants.DATASETS_PATH, dataset_name)
        dataset_info = utils.load_dict_from_json(path.join(dataset_path, 'dataset_info.json'))

        human_scanpaths_dir = path.join(dataset_path, dataset_info['scanpaths_dir'])
        dataset_results_dir = path.join(path.join(constants.RESULTS_PATH, dataset_name + '_dataset'))
        if not path.isdir(dataset_results_dir):
            print('No results found for ' + dataset_name + ' dataset')
            continue
        for config_name in listdir(dataset_results_dir):
            config_folder = path.join(dataset_results_dir,config_name)
            if not path.isdir(config_folder):
                continue

            max_scanpath_length = dataset_info['max_scanpath_length']
            # If desired, this number can be less than the total and the same random subset will be used for all models
            number_of_images    = dataset_info['number_of_images']


            if not path.exists(path.join(config_folder, 'Scanpaths.json')):
                for history_folder in listdir(config_folder):
                    save_path = path.join(config_folder, history_folder)
                    if path.isdir(save_path):
                        # Initialize objects
                        multimatch = Multimatch(dataset_name, human_scanpaths_dir, config_folder, number_of_images, compute_multimatch)

                        subjects_cumulative_performance = CumulativePerformance(dataset_name, number_of_images, max_scanpath_length, compute_cumulative_performance)
                        subjects_cumulative_performance.add_human_mean(human_scanpaths_dir, constants.HUMANS_COLOR)

                        human_scanpath_prediction = HumanScanpathPrediction(dataset_name, human_scanpaths_dir, config_folder,  number_of_images, compute_human_scanpath_prediction)

                        # Compute models metrics and compare them with human subjects metrics
                        color_index = 0
                        model_scanpaths_file = path.join(config_folder, history_folder+'/Scanpaths.json')
                        model_scanpaths      = utils.load_dict_from_json(model_scanpaths_file)
                        # Plot just the subjects
                        subjects_cumulative_performance.plot(save_path=path.join(save_path,"Cumulative Performance Humans.png"))
                        subjects_cumulative_performance.add_model(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])

                        subjects_cumulative_performance.save_results(save_path=save_path, filename=constants.FILENAME)

                        # Human multimatch scores are different for each model, since each model uses different image sizes
                        multimatch.load_human_mean_per_image(model_name, model_scanpaths)
                        multimatch.add_model_vs_humans_mean_per_image(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])
                        multimatch.save_results(save_path=save_path, filename=constants.FILENAME)
                        subjects_cumulative_performance.plot(save_path=path.join(save_path,"Cumulative Performance.png"))
                        multimatch.plot(save_path=save_path)

                        #human_scanpath_prediction.compute_metrics_for_model(model_name,nnIBS)

                        #human_scanpath_prediction.add_baseline_models()

                        
                        
                        #human_scanpath_prediction.save_results(save_path=save_path, filename=constants.FILENAME)

                        dataset_results = utils.load_dict_from_json(path.join(save_path, constants.FILENAME))
                        datasets_results[dataset_name] = dataset_results



                        dataset_results_table = utils.create_table(dataset_results)
                        utils.plot_table(dataset_results_table, title=dataset_name + ' dataset', save_path=save_path, filename='Table.png')
            else:
                # Initialize objects
                multimatch = Multimatch(dataset_name, human_scanpaths_dir, config_folder, number_of_images, compute_multimatch)

                subjects_cumulative_performance = CumulativePerformance(dataset_name, number_of_images, max_scanpath_length, compute_cumulative_performance)
                subjects_cumulative_performance.add_human_mean(human_scanpaths_dir, constants.HUMANS_COLOR)

                human_scanpath_prediction = HumanScanpathPrediction(dataset_name, human_scanpaths_dir, config_folder,  number_of_images, compute_human_scanpath_prediction)

                # Compute models metrics and compare them with human subjects metrics
                color_index = 0
                model_scanpaths_file = path.join(config_folder, 'Scanpaths.json')
                model_scanpaths      = utils.load_dict_from_json(model_scanpaths_file)
                # Plot just the subjects
                subjects_cumulative_performance.plot(save_path=path.join(config_folder,"Cumulative Performance Humans.png"))
                subjects_cumulative_performance.add_model(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])
                subjects_cumulative_performance.save_results(save_path=config_folder, filename=constants.FILENAME)

                # Human multimatch scores are different for each model, since each model uses different image sizes
                multimatch.load_human_mean_per_image(model_name, model_scanpaths)
                multimatch.add_model_vs_humans_mean_per_image(model_name, model_scanpaths, constants.MODELS_COLORS[color_index])
                multimatch.save_results(save_path=config_folder, filename=constants.FILENAME)
                subjects_cumulative_performance.plot(save_path=path.join(config_folder,"Cumulative Performance.png"))
                multimatch.plot(save_path=config_folder)

                human_scanpath_prediction.compute_metrics_for_model(model_name,nnIBS)

                human_scanpath_prediction.add_baseline_models()

                
                
                human_scanpath_prediction.save_results(save_path=config_folder, filename=constants.FILENAME)

                dataset_results = utils.load_dict_from_json(path.join(config_folder, constants.FILENAME))
                datasets_results[dataset_name] = dataset_results



                dataset_results_table = utils.create_table(dataset_results)
                utils.plot_table(dataset_results_table, title=dataset_name + ' dataset', save_path=config_folder, filename='Table.png')    

    if compute_cumulative_performance and compute_multimatch and compute_human_scanpath_prediction:
        final_table = utils.average_results(datasets_results, save_path=constants.RESULTS_PATH, filename='Scores.json')
        utils.plot_table(final_table, title='Ranking', save_path=constants.RESULTS_PATH, filename='Ranking.png')

if __name__ == "__main__":
    available_datasets = utils.get_dirs(constants.DATASETS_PATH)
    
    parser = argparse.ArgumentParser(description='Compute all metrics for each visual search model on each dataset. To only run a subset of them, specify by argument which ones.')
    parser.add_argument('--d', '--datasets', type=str, nargs='*', default=available_datasets, help='Names of the datasets on which to compute the metrics. \
        Values must be in list: ' + str(available_datasets))

    parser.add_argument('--perf', '--performance', action='store_true', help='Compute cumulative performance')
    parser.add_argument('--mm', '--multimatch', action='store_true', help='Compute multimatch on the models')
    parser.add_argument('--hsp', '--human_scanpath_prediction', action='store_true', help='Compute human scanpath prediction on the models. \
        See "Kümmerer, M. & Bethge, M. (2021), State-of-the-Art in Human Scanpath Prediction" for more information. WARNING: If not precomputed, EXTREMELY SLOW!')

    args = parser.parse_args()


    invalid_datasets = not all(dataset in available_datasets for dataset in args.d)
    if (not args.d or invalid_datasets):
        raise ValueError('Invalid set or datasets')

    # If none were specified, default to True
    if not (args.perf or args.mm or args.hsp):
        args.perf = True
        args.mm   = True
        args.hsp  = True

    main(args.d, args.perf, args.mm, args.hsp)
