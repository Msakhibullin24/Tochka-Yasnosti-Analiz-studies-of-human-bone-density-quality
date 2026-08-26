from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS
from typing import Dict, Optional, Any
from mmcv.image import imread
import pandas as pd
import numpy as np
import sys
import os


def define_paths(study: str) -> Dict[str, Optional[str]]:
    """
    Defines the paths and column names used to identify each image uniquely within a study.
    The function provides study-specific folder names, filenames, and metadata. This is a demo only,
    this code should be adapted to the studies which you're examining.

    Args:
        study (str): The name of the study for which paths and column details are required.

    Returns:
        Dict[str, Optional[str]]: A dictionary containing the source folder, filename column,
                                  site column, metadata file, and destination folder.

    Raises:
        ValueError: If the provided study is unknown.
    """
    if study.lower() == 'demo':
        return {'source_folder' : 'images',
                'filename_column' : 'filename',
                'site_column' : None,
                'metadata_file' : 'sample_point_placement.csv',
                'destination_folder' :  'points'}
    else:
        raise ValueError('Unknown study')


def write_pts_file(points: np.ndarray, filename: str,
                   destination_folder: str) -> None:
    """
    Writes 105 points from a given array into a .pts file with the format required by UoMaPm.

    Args:
        points (np.ndarray): A 2D array of shape (1, 105, 2) containing the points to write.
        filename (str): The base name of the file (without extension).
        destination_folder (str): The directory where the .pts file should be saved.

    Returns:
        None
    """
    # Opening the file for writing
    with open(os.path.join(destination_folder, filename) + '.pts', 'w') as f:
        # Writing header information
        f.write("version: 1\n")
        f.write("n_points: 105\n")
        f.write("{\n")

        # Writing each point
        for i in range(105):
            x, y = points[0, i]
            f.write("{:.6f} {:.6f}\n".format(x, y))

        # Closing the file
        f.write("}")


def visualization(results: Dict, image_path: str):
    """
    Visualizes the results on an image and saves the output.
    Args:
        results (Dict): A dictionary containing the data sample
        results to be visualized.
        image_path (str): The file path of the image to be processed.

    Returns:
        None: The function saves the visualized image to the disk and does
        not return anything.

    """
    img = imread(image_path, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_pred=True,
        draw_bbox=False,
        kpt_thr=0.0,
        draw_heatmap=False,
        show_kpt_idx=False,
        out_file=os.path.basename(image_path)[:-4] + '_vis.jpg')


def predict_and_write(row: pd.Series, metadata: Dict[str, Optional[str]],
                      pose_model: Any, visualize: bool) -> None:
    """
    Predicts keypoints and writes them to a .pts file.

    This function is designed to be applied to each row of a pandas DataFrame,
    with relevant fields defined by the
    define_paths helper function.

    Args:
        row (pd.Series): A row from a pandas DataFrame containing metadata
        for the image.
        metadata (Dict[str, Optional[str]]): A dictionary containing paths and
        column names used for processing.
        pose_model (Any): The pose estimation model used for inference.

    Returns:
        None
    """
    image_path = os.path.join(metadata['source_folder'],
                              row[metadata['filename_column']])
    # image location
    try:
        results = inference_topdown(pose_model, image_path)
        # generate prediction

        if visualize:
            results_vis = merge_data_samples(results)
            visualization(results_vis, image_path)

    except FileNotFoundError as e:
        print(e)  # catch problems with model inference if they occur
        return

    points = results[0].pred_instances.keypoints
    destination_image_path = os.path.basename(image_path)[:-4]

    # if we need more than the filename to be unique
    # this is a quirk of Hologic filenames being unique
    # per-system, but not between systems.
    if metadata['site_column'] is not None:
        destination_folder = os.path.join(metadata['destination_folder'],
                                          str(row[metadata['site_column']]))
    else:
        destination_folder = metadata['destination_folder']

    # write predicted keypoints in format needed by UoMaPm
    write_pts_file(points, destination_image_path, destination_folder)


if __name__ == '__main__':
    gpu = int(sys.argv[2])
    study = sys.argv[1]
    visualization_bool = bool(int(sys.argv[3]))

    study_data = define_paths(study)

    model_cfg = 'dxa_points_config.py'
    ckpt = 'checkpoint.pth'
    device = 'cuda:{}'.format(gpu)

    model = init_model(model_cfg, ckpt, device=device)

    if visualization:
        model.cfg.visualizer.radius = 3
        model.cfg.visualizer.alpha = 0.8
        model.cfg.visualizer.line_width = 1

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.set_dataset_meta(model.dataset_meta)

    df = pd.read_csv(study_data['metadata_file'])
    df.apply(lambda x: predict_and_write(x, study_data,
                                         model, visualization_bool), axis=1)
