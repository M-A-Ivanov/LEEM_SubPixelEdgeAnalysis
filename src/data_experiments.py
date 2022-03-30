import json
import os
import pickle
from typing import List

import numpy as np

from edge_processing import EdgeAnalyzer
from global_paths import SRC_FOLDER, REGION, RESULTS_FOLDER, EDGE, TARGET_FOLDER, load_pickle
from edge_segmentation import EdgeContainer
from image_processing import ImageProcessor, average_images, average_image_list
from raw_reader import RawReader
from tqdm import tqdm

from systematic_stuff.fluctuations.boundary_analysis import create_positions, distribution_analysis, fft_analysis, \
    get_correlations, draw_analyzed_edge, get_paper_results
from systematic_stuff.fluctuations.boundary_detection import FluctuationsDetector


def experimental_detection():
    average = 5
    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.align(preprocess=True)
    if average == 1:
        image = reader.read_single(os.path.join(SRC_FOLDER, REGION)).data
    else:
        images = [img.data for img in reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=(50, 50+average))]
        image = average_images(images)
    processor.load_image(image)
    processor.align(preprocess=True)
    mask = processor.get_mask(15)
    mask = processor.cut_to_mask(mask)
    processor.global_hist_equal()
    processor.denoise_nlm(fast=True)
    try:
        processor.canny_edges(sigma=2.25, mask=mask)
        processor.clean_up(25)
    except Exception as e:
        print("Canny fails with: {}".format(e))
    processor.figure_all()


def load_experimental_images(folder_src, num_images=None):
    reader = RawReader()
    if num_images is None:
        list_of_images = reader.read_folder(folder_src)
    else:
        list_of_images = reader.read_single(folder_src, frames=(0, num_images))
    list_of_images = [image.data for image in list_of_images]
    processor = ImageProcessor()
    processor.load_image(list_of_images[0])
    mask = processor.get_mask(15)
    return list_of_images, mask


def experimental_manual_masking(list_of_images, mask=None, align=True,
                                average=1):
    processor = ImageProcessor()
    list_of_images = average_image_list(list_of_images, average)
    processor.load_image(list_of_images[0])
    if align:
        processor.align(preprocess=True)
    if mask is None:
        original_mask = processor.get_mask(15)
    else:
        original_mask = mask
    grouper_canny = EdgeContainer()
    print("Found images: {}".format(len(list_of_images)))
    for i, image in enumerate(tqdm(list_of_images)):
        try:
            processor.load_image(image)
            if align:
                processor.align(preprocess=True)
            mask = processor.cut_to_mask(original_mask)
            processor.global_hist_equal()
            processor.denoise_nlm(fast=True)
            processor.canny_edges(mask=mask, sigma=2.5)
            processor.clean_up(25)
            grouper_canny.append_edges(processor.edge_result())
        except Exception as e:
            print("Frame {} failed with: {}".format(i, e))

    return grouper_canny.get_coordinates()


class ExperimentalFluctDetector(FluctuationsDetector):
    def canny_devernay_many_sigma(self, sigmas: List, folders: List, num_images=None):
        assert len(sigmas) == len(folders)
        list_of_images = self.load_data(num_images)
        self.image_processor.load_image(list_of_images[0].data)
        self.image_processor.align(preprocess=False)
        mask = self.prepare_masks(True)[0]
        groupers_canny = [EdgeContainer() for _ in folders]
        print("Found images: {}".format(len(list_of_images)))
        for i, image in enumerate(tqdm(list_of_images)):
            self.image_processor.load_image(image.data)
            self.image_processor.align(preprocess=False)
            cut_mask = self.image_processor.cut_to_mask(mask)
            for sigma, grouper, folder in zip(sigmas, groupers_canny, folders):
                self.set_parameters(sigma, self.mask_size)
                try:
                    coordinates = self.canny_detection_step(cut_mask)
                    grouper.append_edges(coordinates)
                except Exception as e:
                    self.image_processor.revert_to(3)
                    print("Sigma={} : Frame {} failed with: {}".format(sigma, i, e))
        for folder, grouper_canny in zip(folders, groupers_canny):
            path = os.path.join(self.target_folder, self.edges_names[0], folder)
            os.makedirs(path, exist_ok=True)
            grouper_canny.save_coordinates(path, suffix="canny_devernay")


def modified_create_positions(results_path, picklepath_perp):
    main_coords = os.path.join(results_path, "coordinates_canny_devernay.pickle")
    c = load_pickle(main_coords)

    anal = EdgeAnalyzer(c, low_memory=True)
    with open(picklepath_perp, "rb") as f:
        anal.perpendiculars_to_load(pickle.load(f), low_memory=True)
    positions, info = anal.get_edge_variations(savefig=None)
    edge_length = anal.get_length()
    picklepath = os.path.join(results_path, "offsets_original.pickle")
    with open(picklepath, "wb") as f:
        pickle.dump((positions, edge_length), f)
    infopath = os.path.join(results_path, "detection_info.json")
    with open(infopath, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def run_over_sigmas(do_detection: bool = True, do_transform: bool = True, do_analysis: bool = True):
    sigmas = list(np.arange(2.5, .75, -0.25))
    folders = ["nodenoise and sigma {}".format(s) for s in sigmas]
    common_path = os.path.join(RESULTS_FOLDER, REGION, EDGE[0])
    picklepath_perp = os.path.join(common_path, "perpendiculars.pickle")
    if do_detection:
        image_to_coords = ExperimentalFluctDetector(os.path.join(SRC_FOLDER, REGION), TARGET_FOLDER, EDGE)
        image_to_coords.canny_devernay_many_sigma(sigmas, folders)
    if do_transform:
        for folder in folders:
            res_path = os.path.join(common_path, folder)
            modified_create_positions(results_path=res_path, picklepath_perp=picklepath_perp)
    if do_analysis:
        for folder in folders:
            res_path = os.path.join(RESULTS_FOLDER, REGION, EDGE[0], folder)
            sigma = distribution_analysis(res_path, adjusted=False)
            beta = fft_analysis(res_path, adjusted=False)
            get_correlations(res_path, fps=15, adjusted=False)
            get_paper_results(res_path, beta=beta, sigma=sigma)


if __name__ == "__main__":
    run_over_sigmas(True, True, True)