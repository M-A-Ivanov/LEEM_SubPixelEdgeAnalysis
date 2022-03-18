import math
import os
from abc import ABC
from typing import List

from skimage.color import rgb2gray
from weno4 import weno4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import skimage as io
from scipy import optimize
from skimage.morphology import disk
from skimage.filters import rank, threshold_otsu
from skimage import exposure, img_as_float, img_as_ubyte
from skimage import feature
from skimage import filters
from skimage import transform
from skimage import draw
from skimage import restoration
from skimage import morphology
from skimage import measure
from skimage import util
import os

from PIL import Image
from skimage.restoration import estimate_sigma
from skimage.transform import resize

from canny_devernay import canny_devernay
from edge_segmentation import ImageMasker
from image_recorder import ImageRecorder
from raw_reader import RawReader
from global_paths import SRC_FOLDER


def average_image_list(images, average):
    new_list = []
    while len(images) >= average:
        new_list.append(average_images(images[:average]))
        del images[:average]
    return new_list


def average_images(to_average: list):
    return np.sum(np.array(to_average), axis=0) / len(to_average)


class BackgroundRemover:
    def __init__(self):
        path = os.path.join(SRC_FOLDER, "background.dat")
        self.background = RawReader.read(path).data
        self.background = self.background[192:-192, 192:-192]
        self.background = self.background / np.max(self.background)

    def remove(self, image):
        return image / self.background


class ImageProcessor(ImageRecorder):
    def __init__(self):
        self.images = list()
        self.titles = list()
        self.edges = None
        self.lines = None
        self.align_template = None
        self.cut_corrections = np.array([0, 0])
        self.masker = ImageMasker()

    def load_image(self, img, label=None):
        # if img.shape[-1] == 3:
        #     img = rgb2gray(img)
        self.images = [img_as_float(img)]
        if label is not None:
            self.titles = ["Original at {}".format(label)]
        else:
            self.titles = ['Original']

    def result(self):
        return self.images[-1]

    def edge_result(self):
        return self.edges

    def revert(self, n_steps=1):
        for step in range(n_steps):
            del self.images[-1]
            del self.titles[-1]

    def to_uint8(self):
        self.images.append(img_as_ubyte(self.images[-1]))
        self.titles.append("As uint8")

    def plot(self, figure, savefig):
        if savefig is None:
            plt.show()
        else:
            self.save(savefig, figure)

    def figure_all(self, steps: List = None):
        if steps is not None:
            images = [self.images[i] for i in steps]
            titles = [self.titles[i] for i in steps]
        else:
            images = self.images
            titles = self.titles
        if len(images) == 1:
            return self.figure_initial()
        if len(images) >= 4:
            n = int(np.ceil(np.sqrt(len(images))))
            fig, axes = plt.subplots(n, n, figsize=(18, 9))
            for i in range(n * n):
                axes.flatten()[i].axis("off")
        else:
            fig, axes = plt.subplots(len(images), 1)
        for i, (image, title) in enumerate(zip(images, titles)):
            axes.flatten()[i].imshow(image, cmap='gray')
            axes.flatten()[i].set_title(title, fontsize=8)

        plt.tight_layout()
        return fig

    def plot_all(self, steps: list = None, savefig=None):
        self.plot(self.figure_all(steps), savefig)

    def figure_result(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 9),
                                       sharex='all', sharey='all')

        ax1.imshow(self.images[0], cmap='gray')
        ax1.axis('off')
        ax1.set_title(self.titles[0], fontsize=12)

        ax2.imshow(self.images[-1], cmap='gray')
        ax2.axis('off')
        ax2.set_title(self.titles[-1], fontsize=12)
        return fig

    def plot_result(self, savefig=None):
        self.plot(self.figure_result(), savefig)

    def figure_initial(self):
        fig, ax = plt.subplots()
        ax.imshow(self.images[0], cmap='gray')
        ax.axis('off')
        ax.set_title(self.titles[0])
        return fig

    def save_last(self, savefig=None):
        self.save(savefig, self.images[-1])

    @staticmethod
    def _apply_mask(image, mask):
        new_image = image.copy()
        new_image[np.where(mask == 0)] = 0
        return new_image

    @staticmethod
    def coordinates_to_boolean_image(original_image, coordinates):
        boolean_img = np.zeros_like(original_image, dtype=bool)
        if not isinstance(coordinates, int):
            coordinates = np.rint(coordinates).astype(int)
        boolean_img[coordinates[:, 0], coordinates[:, 1]] = True
        return boolean_img

    @staticmethod
    def boolean_image_to_coordinates(bool_img):
        coordinates = []
        for regionprop in measure.regionprops(measure.label(bool_img)):
            coordinates.append(regionprop.coords)
        if len(coordinates) > 1:
            coordinates = [item for sublist in coordinates for item in sublist]
        return np.squeeze(np.array(coordinates))

    def get_mask(self, line_width=20):
        self.masker.load_image(self.images[-1])
        mask = self.masker.get_manual_mask(line_width)
        return mask

    def cut_to_mask(self, mask):
        masked_image = self.images[-1].copy()
        coordinates = np.ix_(mask.any(1), mask.any(0))
        masked_image = masked_image[coordinates]
        self.images.append(masked_image)
        self.titles.append("Cropped Mask Image")
        self.cut_corrections[0] = np.min(coordinates[0])
        self.cut_corrections[1] = np.min(coordinates[1])
        return mask[coordinates]

    def cut_to_global_coordinates(self, coordinates):
        return np.add(coordinates, self.cut_corrections)

    def global_to_cut_coordinates(self, coordinates):
        return np.subtract(coordinates, self.cut_corrections)

    def invert(self):
        self.images.append(util.invert(image=self.images[-1]))
        self.titles.append("Inverted")

    def align(self, preprocess=True):
        image_fraction = 3 / 5
        crop_x = int(.5 * image_fraction * self.images[-1].shape[0])
        crop_y = int(.5 * image_fraction * self.images[-1].shape[1])
        if preprocess:
            image = restoration.denoise_bilateral(exposure.equalize_hist(self.images[-1]), bins=100)
        else:
            image = self.images[-1]

        if self.align_template is None:
            self.align_template = image[crop_x:-crop_x, crop_y:-crop_y]
        else:
            matched = feature.match_template(image, self.align_template)
            x, y = np.unravel_index(np.argmax(matched), matched.shape)[::-1]
            transformation = transform.SimilarityTransform(translation=(-(crop_x - x), -(crop_y - y)))
            self.images.append(transform.warp(self.images[-1], transformation,
                                              mode='wrap', preserve_range=True).astype(self.images[-1].dtype))
            self.titles.append("Aligned")

    def upscale(self, factor: float or int, mask=None):
        self.images.append(
            resize(self.images[-1], (factor * self.images[-1].shape[0], factor * self.images[-1].shape[1])))
        self.titles.append("Upscaled")
        if mask is not None:
            return resize(mask, (factor * mask.shape[0], factor * mask.shape[1]), order=0).astype(int)

    def find_contours(self, show=True, savefig=None):
        contours = measure.find_contours(self.images[-1], threshold_otsu(self.images[-1]))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(self.images[-1], cmap="gray")
        ax[0].set_title(self.titles[0], fontsize=12)
        ax[1].imshow(self.images[-1], cmap="gray")
        for contour in contours:
            if len(contour) > 70:
                ax[1].plot(contour[:, 1], contour[:, 0], linewidth=1)

        ax[0].axis('image')
        ax[1].axis('image')
        ax[0].set_xticks([])
        ax[1].set_yticks([])
        ax[0].set_xticks([])
        ax[1].set_yticks([])
        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)
        plt.close()
        # self.images.append(image)
        # self.titles.append("Found Contours")

    def canny_edges(self, mask=None, sigma=1):
        otsu = threshold_otsu(self.images[-1])
        self.images.append(feature.canny(self.images[-1],  # low_threshold=0.2*otsu, high_threshold=otsu,
                                         sigma=sigma, mask=mask))
        self.edges = self.cut_to_global_coordinates(self.boolean_image_to_coordinates(self.images[-1]))
        self.titles.append("Detected Edges, Canny")

    def canny_devernay_edges(self, mask=None, sigma=1):
        otsu = threshold_otsu(self.images[-1])
        bool_mask, coordinates = canny_devernay(self.images[-1], low_threshold=0.5 * otsu, high_threshold=otsu,
                                                sigma=sigma, mask=mask)
        self.images.append(bool_mask)
        self.edges = self.cut_to_global_coordinates(self.boolean_image_to_coordinates(self.images[-1]))
        self.titles.append("Detected Edges, Canny-Devernay")

        return self.cut_to_global_coordinates(coordinates)

    def sobel_edges(self, mask=None):
        image = self.images[-1]
        if mask is not None:
            image = self._apply_mask(image, mask)
        self.images.append(filters.sobel(image))
        self.edges = self.boolean_image_to_coordinates(self.images[-1])
        self.titles.append("Detected Edges, Sobel")

    def scharr_edges(self, mask=None):
        image = self.images[-1]
        if mask is not None:
            image = self._apply_mask(image, mask)
        self.images.append(filters.scharr(image))
        self.edges = self.boolean_image_to_coordinates(self.images[-1])
        self.titles.append("Detected Edges, Scharr")

    def prewitt_edges(self, mask=None):
        image = self.images[-1]
        if mask is not None:
            image = self._apply_mask(image, mask)
        self.images.append(filters.prewitt(image))
        self.edges = self.boolean_image_to_coordinates(self.images[-1])
        self.titles.append("Detected Edges, Prewitt")

    # def subpixel_edges(self, mask=None):
    #     image = self.images[-1]
    #     if mask is not None:
    #         image = self._apply_mask(image, mask)
    #     otsu = threshold_otsu(image)
    #     detection = subpixel_edges(image, 0.05 * otsu, 100, 2)
    #     edges = np.stack((detection.y, detection.x), axis=1)
    #     # self.edges = max(np.split(edges, np.where(np.abs(np.hypot(edges[:-1], edges[1:])) > 2)[0]), key=len)
    #     self.images.append(self.coordinates_to_boolean_image(self.images[-1], edges))
    #     self.edges = self.cut_to_global_coordinates(edges)
    #     self.titles.append("Detected Edges, Pino")

    def tanh_edges(self, mask=None):
        def fit_tanh(profile: np.ndarray) -> float:
            def tanh(x, a, eta, phi, b):
                return a * np.tanh(eta * (x + phi)) + b

            x_pixel = np.arange(0, len(profile))
            x_fine = np.linspace(0, len(profile), 10000)
            try:
                fit_params, var_matrix = optimize.curve_fit(tanh, x_pixel, profile)

                return x_fine[np.argmax(np.absolute(np.gradient(tanh(x_fine, *fit_params))))]
            except Exception:
                return 0

        image = self.images[-1]
        if mask is not None:
            image = self._apply_mask(image, mask)
        h, w = image.shape
        coordinates = np.array([[i, fit_tanh(image[:, i])] for i in range(w)])
        coordinates = coordinates[~(coordinates[:, 1] == 0)]
        self.edges = coordinates
        self.images.append(self.coordinates_to_boolean_image(self.images[-1], coordinates))
        self.titles.append("Detected Edges, Tanh")

    def clean_up(self, min_size=25):
        local_coordinates = self.global_to_cut_coordinates(self.edges)
        boolean_image = self.coordinates_to_boolean_image(self.images[-1], local_coordinates)
        clean_boolean_image = morphology.remove_small_objects(boolean_image, min_size, connectivity=2)
        self.images.append(clean_boolean_image)
        self.edges = self.cut_to_global_coordinates(self.boolean_image_to_coordinates(self.images[-1]))
        self.titles.append("Cleaned Up Edge")

    def clean_up_coordinates(self, coordinates, min_size=25):
        local_coordinates = self.global_to_cut_coordinates(coordinates)
        boolean_image = self.coordinates_to_boolean_image(self.images[-1], local_coordinates)
        clean_boolean_image = morphology.remove_small_objects(boolean_image, min_size, connectivity=2)
        self.images.append(clean_boolean_image)
        if np.array_equal(boolean_image, clean_boolean_image):
            self.titles.append("Cleaned Up Edge")
            return coordinates
        boolean_of_rubbish = boolean_image * ~clean_boolean_image
        rubbish_coordinates = self.boolean_image_to_coordinates(boolean_of_rubbish)
        lc_int = np.rint(local_coordinates).astype(int)
        # Assuming shape is (N, 2)
        cumdims = (np.maximum(lc_int.max(), rubbish_coordinates.max()) + 1) ** np.arange(2)  # rubbish_c....shape[1]
        cleaned_up_coordinates = local_coordinates[~np.in1d(lc_int.dot(cumdims), rubbish_coordinates.dot(cumdims))]
        cleaned_up_coordinates = self.cut_to_global_coordinates(cleaned_up_coordinates)
        self.edges = cleaned_up_coordinates
        self.titles.append("Cleaned Up Edge")
        return cleaned_up_coordinates

    def open(self, disk_size=None):
        if disk_size is None:
            disk_size = 3
        selem = morphology.selem.disk(disk_size)
        self.images.append(morphology.binary_opening(self.images[-1], selem=selem))
        self.titles.append("Opended Edges")

    def close(self, disk_size=None):
        if disk_size is None:
            disk_size = 3
        selem = morphology.selem.disk(disk_size)
        self.images.append(morphology.binary_closing(self.images[-1], selem=selem))
        self.titles.append("Closed Edges")

    def erode(self, disk_size=None):
        if disk_size is None:
            disk_size = 3
        selem = morphology.selem.disk(disk_size)
        self.images.append(morphology.binary_erosion(self.images[-1], selem=selem))
        self.titles.append("Eroded Edges")

    def dilate(self, disk_size=None):
        if disk_size is None:
            disk_size = 3
        selem = morphology.selem.disk(disk_size)
        self.images.append(morphology.binary_dilation(self.images[-1], selem=selem))
        self.titles.append("Closed Area Edges")

    def local_hist_equal(self, disk_size=15):
        selem = io.morphology.disk(disk_size)
        self.images.append(io.filters.rank.equalize(self.images[-1], selem=selem))
        self.titles.append("Locally Equalized Hist")

    def global_hist_equal(self, mask=None):
        self.images.append(exposure.equalize_hist(self.images[-1], mask=mask))
        self.titles.append("Globally Equalized Hist")

    def clahe_hist_equal(self, kernel_size=None):
        if kernel_size is not None:
            self.images.append(exposure.equalize_adapthist(self.images[-1], kernel_size=kernel_size))
        else:
            self.images.append(exposure.equalize_adapthist(self.images[-1]))
        self.titles.append("CLAHE Equalized Hist")

    def estimate_noise(self):
        return estimate_sigma(self.images[-1])

    def denoise_boxcar(self, kernel_radius=1):
        footprint = disk(kernel_radius)
        self.images.append(img_as_float(rank.mean(self.images[-1], footprint)))
        self.titles.append("Boxcar Denoising")

    def denoise_gaussian(self, sigma=4):
        self.images.append(filters.gaussian(self.images[-1], sigma=sigma))
        self.titles.append("Gaussian Denoising")

    def denoise_bilateral(self, bins=None):
        if bins is None:
            bins = 10000
        else:
            bins = int(bins)
        self.images.append(restoration.denoise_bilateral(self.images[-1], bins=bins))
        self.titles.append("Bilateral Denoising")

    def denoise_wavelet(self):
        self.images.append(restoration.denoise_wavelet(self.images[-1]))
        self.titles.append("Wavelet Denoising")

    def denoise_tv(self):
        self.images.append(restoration.denoise_tv_chambolle(self.images[-1], weight=1e-1))
        self.titles.append("TV Denoising")

    def denoise_nlm(self, fast=True):
        sigma = estimate_sigma(self.images[-1])
        if fast:
            h = 0.8 * sigma
        else:
            h = 0.6 * sigma
        self.images.append(restoration.denoise_nl_means(self.images[-1], sigma=sigma, h=h,
                                                        fast_mode=fast))
        self.titles.append("NLM Denoising")

    def get_probabalistic_hough(self):
        return io.transform.probabilistic_hough_line(self.images[-1])

    def find_probabalistic_hough(self):
        lines = self.get_probabalistic_hough()
        lined_image = self.images[0].copy()
        for line in lines:
            p0, p1 = line
            x_coords, y_coords = io.draw.line(p0[0], p0[1], p1[0], p1[1])
            lined_image[x_coords, y_coords] = 60
        self.images.append(lined_image)
        self.titles.append("Found_lines")

    def get_linear_hough(self):
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 90, endpoint=False)
        return transform.hough_line(self.images[-1], theta=tested_angles)

    def find_linear_hough(self):
        out, theta, d = self.get_linear_hough()
        self.images.append(out)
        # lined_image = self.images[0].copy()
        # for _, angle, dist in zip(*transform.hough_line_peaks(out, theta, d)):
        #     (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #     ax[2].axline((x0, y0), slope=np.tan(angle + np.pi / 2))

    def get_profile(self, pt1: np.ndarray, pt2: np.ndarray):
        # turning points around, since profile line takes (col, row)
        return measure.profile_line(self.images[-1], pt1[::-1], pt2[::-1], mode='reflect')
