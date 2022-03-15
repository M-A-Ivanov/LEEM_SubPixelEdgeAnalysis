import os
from typing import List

from matplotlib import pyplot as plt
from scipy import optimize

from global_parameters import CANNY_SIGMA, MASK_SIZE
from global_paths import SRC_FOLDER, TARGET_FOLDER, REGION, save_pickle, load_pickle, EDGE
from edge_segmentation import EdgeContainer, ImageMasker
from image_processing import ImageProcessor, average_images
from raw_reader import RawReader
import numpy as np
from tqdm import tqdm
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)


def try_canny_detection():
    edge = "edge 15"
    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.align(preprocess=True)

    image = reader.read_single(os.path.join(SRC_FOLDER, REGION)).data

    processor.load_image(image)

    processor.align(preprocess=True)
    if edge:
        load_pickle(os.path.join(TARGET_FOLDER, edge, "mask.pickle"))
    mask = processor.get_mask(5)
    mask = processor.cut_to_mask(mask)
    processor.clahe_hist_equal()
    processor.denoise_nlm(fast=True)
    processor.canny_edges(sigma=0.5, mask=mask)
    # processor.subpixel_edges(mask=None)
    processor.clean_up(5)
    processor.figure_all()


def try_canny_devernay_detection(edge=None):
    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.align(preprocess=False)

    image = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data

    processor.load_image(image)

    processor.align(preprocess=False)
    if edge:
        mask = load_pickle(os.path.join(TARGET_FOLDER, edge, "mask.pickle"))
    else:
        mask = processor.get_mask(MASK_SIZE)
    local_mask = processor.cut_to_mask(mask)
    processor.clahe_hist_equal()
    processor.denoise_nlm(fast=True)
    coords = processor.canny_devernay_edges(sigma=CANNY_SIGMA, mask=local_mask)
    coords = processor.clean_up_coordinates(coords, 5)
    processor.plot_all()
    processor.revert(5)


def try_tanh_detection():
    def tanh(x, a, eta, phi, b):
        return a * np.tanh(eta * (x + phi)) + b

    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.align(preprocess=True)

    image = reader.read_single(os.path.join(SRC_FOLDER, REGION)).data

    processor.load_image(image)

    processor.align(preprocess=False)
    mask = processor.get_mask(10)
    mask = processor.cut_to_mask(mask)
    processor.clahe_hist_equal()
    processor.denoise_nlm(fast=True)
    image = processor.result()
    plt.imshow(image * mask, cmap="gray")
    plt.show()
    edge = np.zeros((image.shape[1], 2))
    for x_coordinate in range(image.shape[1]):
        profile_line = image[:, x_coordinate]
        y_pixel = np.arange(profile_line.size)
        profile_line = profile_line[mask[:, x_coordinate] != 0]
        if profile_line.size < 10:
            continue
        y_pixel = y_pixel[mask[:, x_coordinate] != 0]
        y_fine = np.linspace(y_pixel[0], y_pixel[-1], 1000)
        try:
            fit_params, var_matrix = optimize.curve_fit(tanh, y_pixel, profile_line)
            edge[x_coordinate, 0] = x_coordinate
            edge[x_coordinate, 1] = y_fine[np.argmax(np.absolute(np.gradient(tanh(y_fine, *fit_params))))]
            plt.plot(y_pixel, profile_line)
            plt.plot(y_fine, tanh(y_fine, *fit_params), 'r--')
            plt.show()
            plt.close()
        # except OptimizeWarning as e:
        except RuntimeWarning as e:
            print("error was {}".format(e))
            plt.plot(profile_line)
            plt.show()
            plt.close()
    print(edge)


class FluctuationsDetector:
    def __init__(self, folder_src: str, target_folder: str, edges: List):
        self.folder_src = folder_src
        self.target_folder = target_folder
        self.edges_names = edges
        self.image_processor = ImageProcessor()

    def prepare_detector(self):
        reader = RawReader()
        image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
        self.image_processor.load_image(image_align)
        self.image_processor.align(preprocess=False)

    def get_masks(self, load=True, save=True, mask_size=5):
        if load:
            masks = self._load_masks()
        else:
            masks = [self.image_processor.get_mask(mask_size) for _ in self.edges_names]
            if save:
                self._save_masks(masks)
        return masks

    def load_data(self, num_images=None):
        reader = RawReader()
        if num_images is None:
            return reader.read_folder(self.folder_src)
        else:
            return reader.read_single(self.folder_src, frames=(0, num_images))

    def _save_masks(self, masks):
        for (mask, edge) in zip(masks, self.edges_names):
            path = os.path.join(self.target_folder, edge)
            save_pickle(os.path.join(path, "mask.pickle"), mask)

    def _load_masks(self):
        return [load_pickle(os.path.join(self.target_folder, edge, "mask.pickle")) for edge in self.edges_names]

    def many_edge_canny_devernay_detection(self, load_masks: bool = True, num_images: int = None):
        """Algorithm 1: Choosing the location of step/boundary. This algorithm:
                            * Loads raw .dat images from 'folder_src'
                            * For the first image loaded:
                                - Saves alignment template (middle 1/2 of image area)
                                - Lets user choose the boundary to isolate by drawing a mask through clicks
                                - Save masks as binary arrays
                            * For every image loaded:
                                - Align using template
                                - Zoom into mask
                                - "Globally" equalize histogram (practically, that's local)
                                - Denoise image (usually, using Non-local means, but other methods available)
                                - Detect edge using Canny edge & collect coordinates
                            * Save all coordinates in shape [frames, array(n_points, 2)]
                                N.B. Array of coordinates is arranged like : (y-coordinate, x-coordinate)
        """
        list_of_images = self.load_data(num_images)
        self.image_processor.load_image(list_of_images[0].data)
        self.image_processor.align(preprocess=False)
        if load_masks:
            masks = self._load_masks()
        else:
            masks = [self.image_processor.get_mask(MASK_SIZE) for _ in self.edges_names]
            self._save_masks(masks)
        groupers_canny = [EdgeContainer() for _ in self.edges_names]
        print("Found images: {}".format(len(list_of_images)))
        for i, image in enumerate(tqdm(list_of_images)):
            # time = round((image.metadata['timestamp'] - first_image_time).total_seconds(), 3)
            try:
                self.image_processor.load_image(image.data)
                self.image_processor.align(preprocess=False)
                for original_mask, grouper_canny in zip(masks, groupers_canny):
                    mask = self.image_processor.cut_to_mask(original_mask)
                    self.image_processor.clahe_hist_equal()
                    self.image_processor.denoise_nlm(fast=True)
                    coordinates = self.image_processor.canny_devernay_edges(mask=mask, sigma=CANNY_SIGMA)
                    coordinates = self.image_processor.clean_up_coordinates(coordinates, 10)
                    grouper_canny.append_edges(coordinates)
                    self.image_processor.revert(5)  # TODO: autoreverter using Enum for processes?
            except Exception as e:
                print("Frame {} failed with: {}".format(i, e))
        for edge, grouper_canny, mask in zip(self.edges_names, groupers_canny, masks):
            print(edge)
            path = os.path.join(self.target_folder, edge)
            os.makedirs(path, exist_ok=True)
            grouper_canny.save_coordinates(path, suffix="canny_devernay")

    def canny_detection(self, load_masks=True, num_images=None):
        """Algorithm 1: Choosing the location of step/boundary. This algorithm:
                            * Loads raw .dat images from 'folder_src'
                            * For the first image loaded:
                                - Saves alignment template (middle 1/2 of image area)
                                - Lets user choose the boundary to isolate by drawing a mask through clicks
                                - Save masks as binary arrays
                            * For every image loaded:
                                - Align using template
                                - Zoom into mask
                                - "Globally" equalize histogram (practically, that's local)
                                - Denoise image (usually, using Non-local means, but other methods available)
                                - Detect edge using Canny edge & collect coordinates
                            * Save all coordinates in shape [frames, array(n_points, 2)]
                                N.B. Array of coordinates is arranged (y-coordinate, x-coordinate)
        """
        mask_size = 12
        list_of_images = self.load_data(num_images)
        self.image_processor.load_image(list_of_images[0].data)
        self.image_processor.align(preprocess=True)
        masks = self.get_masks(load=load_masks, mask_size=mask_size)
        groupers_canny = [EdgeContainer() for _ in self.edges_names]
        print("Found images: {}".format(len(list_of_images)))
        for i, image in enumerate(tqdm(list_of_images)):
            # time = round((image.metadata['timestamp'] - first_image_time).total_seconds(), 3)
            try:
                self.image_processor.load_image(image.data)
                self.image_processor.align(preprocess=True)
                for original_mask, grouper_canny in zip(masks, groupers_canny):
                    mask = self.image_processor.cut_to_mask(original_mask)
                    self.image_processor.global_hist_equal()
                    self.image_processor.denoise_nlm(fast=True)
                    self.image_processor.canny_edges(mask=mask, sigma=0.75)
                    self.image_processor.clean_up(10)
                    grouper_canny.append_edges(self.image_processor.edge_result())
                    self.image_processor.revert(5)
            except Exception as e:
                print("Frame {} failed with: {}".format(i, e))
        for edge, grouper_canny, mask in zip(self.edges_names, groupers_canny, masks):
            print(edge)
            path = os.path.join(self.target_folder, edge)
            os.makedirs(path, exist_ok=True)
            grouper_canny.save_coordinates(path, suffix="canny")

    def simple_tanh_detect(self, num_images=None):
        def tanh(x, a, eta, phi, b):
            return a * np.tanh(eta * (x + phi)) + b

        groupers_canny = [EdgeContainer() for _ in self.edges_names]
        masks = [self.image_processor.get_mask(10) for _ in self.edges_names]
        list_of_images = self.load_data(num_images)
        self.image_processor.load_image(list_of_images[0].data)
        self.image_processor.align(preprocess=True)
        for i, image in enumerate(tqdm(list_of_images)):
            # time = round((image.metadata['timestamp'] - first_image_time).total_seconds(), 3)
            self.image_processor.load_image(image.data)
            self.image_processor.align(preprocess=True)
            for original_mask, grouper_canny in zip(masks, groupers_canny):
                mask = self.image_processor.cut_to_mask(original_mask)
                self.image_processor.global_hist_equal()
                self.image_processor.denoise_nlm(fast=True)
                image = self.image_processor.result()
                edge = np.zeros((image.shape[0], 2))
                for x_coordinate in range(image.shape[0]):
                    profile_line = image[x_coordinate, :]
                    y_pixel = np.arange(profile_line.size)
                    y_fine = np.linspace(y_pixel[0], y_pixel[-1], 10000)
                    try:
                        fit_params, var_matrix = optimize.curve_fit(tanh, y_pixel, profile_line)
                        edge[x_coordinate, 0] = x_coordinate
                        edge[x_coordinate, 1] = y_fine[np.argmax(np.absolute(np.gradient(tanh(y_fine, *fit_params))))]
                    except Exception as e:
                        print("fit failure: {}".format(e))
                grouper_canny.append_edges(edge)
                self.image_processor.revert(5)


class FrontEndDetector(FluctuationsDetector):
    def __init__(self, folder_src: str, target_folder: str, edges: List):
        super(FrontEndDetector, self).__init__(folder_src, target_folder, edges)
        self.flags = {"HistEqual": "CLAHE",
                      "Denoising": "NLM",
                      "CannySigma": 1}

    def denoiser(self):
        flag = self.flags["Denoising"]
        if flag == "NLM":
            return self.image_processor.denoise_nlm(fast=True)
        if flag == "Bilateral Filtering":
            return self.image_processor.denoise_bilateral()
        if flag == "Wavelet":
            return self.image_processor.denoise_wavelet()
        if flag == "TV":
            return self.image_processor.denoise_tv()

    def hist_equalizer(self):
        flag = self.flags["HistEqual"]
        if flag == "CLAHE":
            return self.image_processor.clahe_hist_equal()
        if flag == "Global":
            return self.image_processor.global_hist_equal()

    def single_edge_canny_devernay(self, load_masks: bool = True, save_masks: bool = True, mask_size: int = 8,
                                   num_images: int = None):
        list_of_images = self.load_data(num_images)
        original_mask = self.get_masks(load_masks, save_masks, mask_size)[0]
        figs = []
        for i, image in enumerate(tqdm(list_of_images)):
            # time = round((image.metadata['timestamp'] - first_image_time).total_seconds(), 3)
            self.image_processor.load_image(image.data)
            self.image_processor.align(preprocess=False)
            mask = self.image_processor.cut_to_mask(original_mask)
            self.hist_equalizer()
            self.denoiser()
            figs.append(self.image_processor.figure_all())

        return figs


if __name__ == '__main__':
    # manual_masking(SRC_FOLDER, TARGET_FOLDER, EDGE, load_masks=True, num_images=10)
    try_canny_devernay_detection(edge="edge 1")
