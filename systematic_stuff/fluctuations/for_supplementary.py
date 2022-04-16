import pickle
from typing import List

import cv2 as cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.stats import stats
from skimage.measure import block_reduce
from skimage.transform import resize
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
from matplotlib import colors as c
from matplotlib import cm as cm
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import morphology, img_as_ubyte, measure
from tifffile import tifffile
from tqdm import tqdm

from data_analysis import FFTResults, DistributionResults, ExperimentalDistributionResults, ExperimentalFFTResults
from data_experiments import experimental_manual_masking, load_experimental_images
from edge_processing import EdgeAnalyzer
from global_paths import SRC_FOLDER, REGION, RESULTS_FOLDER
from global_parameters import pixel, AxisNames, CANNY_SIGMA
from image_processing import ImageProcessor, BackgroundRemover
from raw_reader import RawReader
from systematic_stuff.convenience_functions import raw_to_video
from systematic_stuff.fluctuations.boundary_analysis import _all_regions, load_pickle, _all_edges, \
    distribution_analysis, fft_analysis, get_paper_results
import os

video_paths = [os.path.join(SRC_FOLDER, region) for region in _all_regions()]
target_folder = r'F:\cardiff cloud\OneDrive - Cardiff University\UNIVERSITY\PhD\coexistence paper\supplementary'


# target_folder = r'C:\Users\User\OneDrive - Cardiff University\UNIVERSITY\PhD\coexistence paper\supplementary'


class FigureCreator:
    def __init__(self, region, edge):
        self.region = region
        self.res_path = os.path.join(RESULTS_FOLDER, region, edge)
        try:
            self.edge_process = EdgeAnalyzer(load_pickle(os.path.join(self.res_path, "coordinates_canny.pickle")))
        except FileNotFoundError as e:
            self.edge_process = EdgeAnalyzer(
                load_pickle(os.path.join(self.res_path, "coordinates_canny_devernay.pickle")))
        self.coordinates = self.edge_process.edges
        self.read = RawReader()
        self.image_process = ImageProcessor()

        self._prepare_image(frame=0)

        self.images = None

        self.fig, self.ax = plt.subplots()
        self.filename_counter = 0
        self.ax.grid(False)
        plt.tight_layout()

    def _load_images(self, frame=None, n_images=None, every_n_images=None):
        if frame is None:
            self.images = [im.data for im in self.read.read_folder(folder_path=os.path.join(SRC_FOLDER, self.region),
                                                                   first_n=n_images,
                                                                   every_nth=every_n_images)]
        else:
            self.images = self.read.read_single(folder_path=os.path.join(SRC_FOLDER, self.region), frames=frame).data

    def _preprocess_image(self, image):
        self.image_process.load_image(image)
        self.image_process.clahe_hist_equal()
        self.image_process.align(preprocess=False)
        return self.image_process.result()

    def _prepare_image(self, frame=0):
        self._load_images(frame)
        image = self._preprocess_image(self.images)
        return image

    def draw_image(self, frame=0):
        self.ax.imshow(self._prepare_image(frame=frame), cmap="gray")

    def _draw_edge(self, image, edge_coordinates, color, alpha):
        image = np.stack((image,) * 3, axis=-1)
        color = np.asarray(c.to_rgb(color))
        if not isinstance(edge_coordinates, int):
            edge_coordinates = np.rint(edge_coordinates).astype(int)
        image[edge_coordinates[:, 1], edge_coordinates[:, 0]] = (1 - alpha) * image[
            edge_coordinates[:, 1], edge_coordinates[:, 0]] + alpha * color
        return image

    def _prepare_analyzed_edge(self, frame=0, alpha=0.5):
        image = self._prepare_image(frame=frame)
        color_multiplier = np.max(image) / 255
        color = np.array([0, 255, 0]) * color_multiplier
        edge_coords = self.coordinates[frame]
        image[edge_coords[:, 0] - 1, edge_coords[:, 1] - 1] = (1 - alpha) * image[
            edge_coords[:, 0] - 1, edge_coords[:, 1] - 1] + alpha * color

        return image

    def draw_analyzed_edge(self, frame=0, alpha=0.8):
        self.ax.imshow(self._prepare_analyzed_edge(frame=frame, alpha=alpha))

    def _get_perpendiculars(self, load=True):
        if not self.edge_process.perpendiculars:
            if load:
                picklepath_perp = os.path.join(self.res_path, "perpendiculars.pickle")
                with open(picklepath_perp, "rb") as f:
                    self.edge_process.perpendiculars_to_load(pickle.load(f), low_memory=False)
            else:
                self.edge_process.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=16)

    def _select_perpendiculars(self, every_n_perps=0):
        self._get_perpendiculars()
        total_perps = len(self.edge_process.perpendiculars)
        if every_n_perps == 0:
            middle_perp_index = int(total_perps / 2)
            self.edge_process.perpendiculars = [self.edge_process.perpendiculars[int(4 / 3 * middle_perp_index + 2)]]
            return
        self.edge_process.perpendiculars = self.edge_process.perpendiculars[::every_n_perps]

    def draw_perpendiculars(self, alpha, color='red', linestyle="--", line_thickness=3):
        for perp in self.edge_process.perpendiculars:
            # self.ax.scatter(perp.mid[1], perp.mid[0], color='green', alpha=0.5, s=1)  # the midpoints of fit lines
            self.ax.scatter(perp.mid[0], perp.mid[1], color="k", alpha=alpha, s=line_thickness)
            self.ax.axline((perp.mid[0], perp.mid[1]), (perp.second[0], perp.second[1]), alpha=alpha, color=color,
                           linewidth=line_thickness, linestyle=linestyle)

    def show(self):
        plt.show()

    def save(self, suffix, dpi=1024):
        path = os.path.join(self.res_path, "for_suppl")
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "frame_{}.png".format(suffix)), dpi=dpi)
        self.filename_counter += 1

    def show_and_save(self, dpi):
        self.save(dpi)
        self.show()

    def clear(self):
        self.fig, self.ax = plt.subplots()
        self.filename_counter = 0
        self.ax.grid(False)
        plt.axis("equal")
        plt.axis("off")
        plt.tight_layout()

    def add_scalebar(self):
        self.ax.add_artist(
            ScaleBar(pixel, "nm", length_fraction=0.3, width_fraction=0.02, location="lower right", color="goldenrod",
                     frameon=False,
                     font_properties={"size": 30}))

    def _get_rectangle_for_perps(self, distance_between_perpendiculars=10, rotated=True):
        mask_image = np.zeros_like(self.images, dtype=np.uint8)
        for perp in self.edge_process.perpendiculars:
            mask_image[int(perp.mid[1] - 1), int(perp.mid[0] - 1)] = 1
        kernel = np.ones((distance_between_perpendiculars, distance_between_perpendiculars), np.uint8)
        img = cv2.dilate(mask_image, kernel, iterations=1)
        return self._get_rectangle(img, rotated)

    def _get_rectangle(self, mask, rotated: bool):
        contour, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if rotated:
            rect = cv2.minAreaRect(contour[0])
        else:
            x, y, w, h = cv2.boundingRect(contour[0])
            rect = ((float(x + w / 2), float(y + h / 2)), (float(w), float(h)), 0.)
        return rect

    def load_mask(self, edge=None):
        if edge is None:
            path = self.res_path
        else:
            path = os.path.join(RESULTS_FOLDER, self.region, edge)
        return load_pickle(os.path.join(path, "mask.pickle"))

    def _draw_rectangle(self, rect, color="orange", linestyle="--", line_thickness=2., ax=None):
        if ax is None:
            ax = self.ax
        box = cv2.boxPoints(rect)
        for i, j in zip(box, np.roll(box, 1, axis=0)):
            ax.plot([i[0] + 1, j[0] + 1], [i[1] + 1, j[1] + 1], linestyle=linestyle, color=color,
                    linewidth=line_thickness)
        # self.ax.plot([box[-1, 0], box[1, 0]], [box[-1, 1], box[0, 1]], linestyle=linestyle, color=color,
        #              linewidth=line_thickness)

    def zoom_in_rectangle(self, rect):
        box = cv2.boxPoints((rect[0], rect[1], 0.0))
        self.ax.set_xlim(xmin=min(box[:, 0]) - 1, xmax=max(box[:, 0]) + 1)
        self.ax.set_ylim(ymin=max(box[:, 1]) + 1, ymax=min(box[:, 1]) - 1)

    def crop_to_rectangle(self, img, rect):
        # rotate img
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))
        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect0)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0
        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                   pts[1][0]:pts[2][0]]

        return img_crop

    def draw_perpendiculars_detection(self, n_perps=5, frame=0):
        self._select_perpendiculars(3)
        self.draw_perpendiculars(0.90, line_thickness=3, color="red", linestyle='--')
        offsets = self.edge_process.get_edge_variations(savefig=None, frames=frame)

    def draw_figure_1(self, frame=0):
        self.draw_image(frame)
        self._get_perpendiculars()
        rect = self._get_rectangle_for_perps(rotated=False, distance_between_perpendiculars=20)
        self._draw_rectangle(rect, line_thickness=2, color="sandybrown")
        plt.tight_layout()
        # plt.axis("equal")
        plt.axis("off")
        self.ax.set_xlim()
        self.add_scalebar()
        self.save(1)
        self.show()
        self.clear()

    def draw_figure_2(self, frame=0):
        self._load_images(frame)
        self.image_process.load_image(self.images)
        mask = self.load_mask()
        new_mask = self.image_process.cut_to_mask(mask)
        self.image_process.normalize()
        self.image_process.denoise_bilateral()
        edge_coords = self.coordinates[frame]
        edge_coords = np.subtract(edge_coords, np.roll(self.image_process.cut_corrections, 1, axis=0))
        image = self._draw_edge(self.image_process.result(), edge_coordinates=edge_coords, color="lawngreen", alpha=0.4)
        self.ax.imshow(image)
        # self.draw_analyzed_edge(frame, alpha=0.5)
        self._get_perpendiculars()
        for perp in self.edge_process.perpendiculars:
            perp.mid = np.subtract(perp.mid, np.roll(self.image_process.cut_corrections, 1, axis=0))
            perp.second = np.subtract(perp.second, np.roll(self.image_process.cut_corrections, 1, axis=0))
        self._select_perpendiculars(every_n_perps=3)
        self.draw_perpendiculars(alpha=0.3, line_thickness=2, color="darkred")
        self._select_perpendiculars(0)
        self.draw_perpendiculars(alpha=0.9, line_thickness=3, color="red", linestyle='--')
        # rect = self._get_rectangle_for_perps(rotated=False, distance_between_perpendiculars=20)
        self.add_scalebar()
        rect_to_draw = np.asarray(self._get_rectangle_for_perps(rotated=False, distance_between_perpendiculars=32))
        rect_to_draw[2] = np.rad2deg(-np.arctan(self.edge_process.perpendiculars[0].angle_to_y()))
        rect_to_draw = tuple(rect_to_draw)
        self._draw_rectangle(rect_to_draw, line_thickness=2, color="dodgerblue")
        # plt.axis("equal")
        plt.axis("off")
        self.save(2)
        self.show()
        self.clear()

    def draw_figure_3(self, frame=0):
        def rotate(point, origin, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            ox, oy = origin
            px, py = point

            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy) - 0.5
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy) - 0.5
            return qx, qy

        self._load_images(frame)
        self.image_process.load_image(self.images)
        mask = self.load_mask()
        new_mask = self.image_process.cut_to_mask(mask)
        self.image_process.normalize()
        self.image_process.denoise_bilateral(bins=1e7)
        # self.image_process.denoise_gaussian(0.5)

        self._select_perpendiculars(0)
        perp = self.edge_process.perpendiculars[0]
        perp.mid = np.subtract(perp.mid, np.roll(self.image_process.cut_corrections, 1, axis=0))
        perp.second = np.subtract(perp.second, np.roll(self.image_process.cut_corrections, 1, axis=0))
        edge = self.edge_process.edges[frame]
        edge = np.subtract(edge, np.roll(self.image_process.cut_corrections, 1, axis=0))
        # image = self._draw_edge(self.image_process.result(), edge_coordinates=edge, color="lawngreen", alpha=0.4)
        image = self.image_process.result()
        self.add_scalebar()
        # for color, name, detection_method in zip(["deepskyblue", "darkorange"],
        #                                          ["Linear Interpolation", "Weno projection"],
        #                                          [perp.project_interpolation, perp.project_weno]):
        color, name, detection_method = "darkorange", "Weno projection", perp.project_weno
        detection_method(edge)
        original, detection, projected, sign = perp.found_points[-1]
        detection = np.apply_along_axis(rotate, axis=0, arr=detection, origin=np.array([0, 0]),
                                        angle=np.deg2rad(perp.angle_to_y()))
        if original.ndim > 1:
            original = np.apply_along_axis(rotate, axis=1, arr=original, origin=np.array([0, 0]),
                                           angle=np.deg2rad(perp.angle_to_y()))
        else:
            original = np.asarray(rotate(original, origin=np.array([0, 0]), angle=np.deg2rad(perp.angle_to_y())))
        projected = rotate(projected, origin=np.array([0, 0]), angle=np.deg2rad(perp.angle_to_y()))
        #  Minuses are because in CV/skimage, y-axis goes downwards, in calculation, goes upwards
        self.ax.plot(detection[0,], -detection[1,], linewidth=5, linestyle='-', color=color,
                     label=name)
        if original.ndim == 1:
            self.ax.scatter(original[0,], -original[1,], color=color)
        else:
            self.ax.scatter(original[1:-1, 0], -original[1:-1, 1], color=color)
        # self.ax.scatter(projected[0] + .5, -projected[1] + .5, marker="x", color=color, s=50, alpha=1)

        image, extent = perp.get_local_image(image=image, radius=17, offset_to_mid=0,
                                             rotation=False)
        extent[0] -= .5
        extent[1] -= .5
        extent[2] += .5
        extent[3] += .5

        self.ax.imshow(image, extent=extent, cmap="gray")
        xy1, xy2 = np.array([0, 0], dtype=float), np.array([0, 0], dtype=float)
        slope = perp.get_perp_direction()
        xy2[0] = xy1[0] + slope[0]
        xy2[1] = xy1[1] - slope[1]
        # xy2[1] = xy2[1]
        self.ax.axline(xy1=xy1, xy2=xy2,
                       color="red", linewidth=6, linestyle='--')
        # plt.axis("equal")
        # plt.legend(prop={'size': 30})
        plt.axis("off")
        self.save(3)
        self.show()
        self.clear()

    def get_ROI(self):
        mask_image = np.zeros_like(self.images, dtype=np.uint8)
        self._get_perpendiculars()
        for perp in self.edge_process.perpendiculars:
            mask_image[int(perp.mid[1]), int(perp.mid[0])] = 1
        selem = morphology.disk(15)
        mask_image = morphology.binary_dilation(mask_image, selem=selem)
        coordinates = np.ix_(mask_image.any(1), mask_image.any(0))
        cut_corrections = np.array([np.min(coordinates[1]), np.min(coordinates[0])])
        self.image_process.load_image(self.images)
        self.image_process.align(preprocess=True)
        _ = self.image_process.cut_to_mask(mask_image)

        return cut_corrections

    def draw_figure_4(self, frame=0):
        def tanh(x, a, eta, phi, b):
            return a * np.tanh(eta * (x + phi)) + b

        # Load image and get mask based on perpendiculars
        self._load_images(frame)
        cut_corrections = self.get_ROI()
        # Narrow down to 1 perpendicular asap for efficiency
        self._select_perpendiculars(0)
        perp = self.edge_process.perpendiculars[0]
        edge = self.edge_process.edges[frame]

        # Find rough offset (we don't care about adjusting to the mask just yet)
        rough_offset = perp.project_weno(edge)

        # Now we do...
        perp.mid = np.subtract(perp.mid, cut_corrections)
        perp.second = np.subtract(perp.second, cut_corrections)

        # Same process as in computations
        self.image_process.normalize()
        self.image_process.denoise_bilateral()

        x_pixel, x_fine, profile_line = perp.get_profile_around_detection(self.image_process.result(), rough_offset, 8)

        self.ax.scatter(x_pixel, profile_line, color='k')
        fit_params, var_matrix = optimize.curve_fit(tanh, x_pixel, profile_line)
        self.ax.plot(x_fine, tanh(x_fine, *fit_params), color='r')
        # self.ax.axvline(adjustment, linestyle='--', color='r')
        self.ax.axvline(0, linestyle='--', color='sandybrown')
        self.ax.set_xlabel("Position along perpendicular", fontsize=12)
        self.ax.set_ylabel("Pixel intensity", fontsize=12)
        self.ax.set_box_aspect(1)
        plt.tight_layout()

        self.save(4)
        self.show()
        self.clear()

    def draw_figure_denoising(self, frame=0):
        plt.close("all")
        self._load_images(frame)
        self.get_ROI()
        self.image_process.global_hist_equal()
        fig, ax = plt.subplots(nrows=3, ncols=2)
        ax[0, 0].imshow(self.image_process.result(), cmap="gray")
        ax[0, 0].axis("equal")
        ax[0, 0].axis("off")
        ax[0, 0].title.set_text("Original")
        for i, denoising_method in enumerate([self.image_process.denoise_gaussian, self.image_process.denoise_bilateral,
                                              self.image_process.denoise_nlm, self.image_process.denoise_wavelet,
                                              self.image_process.denoise_tv]):
            column = int((i + 1) / 3)
            row = int((i + 1) % 3)
            denoising_method()

            ax[row, column].imshow(self.image_process.result(), cmap="gray")
            ax[row, column].title.set_text(self.image_process.titles[-1])
            ax[row, column].axis("equal")
            ax[row, column].axis("off")
            self.image_process.revert()

        path = os.path.join(self.res_path, "for_suppl")
        # plt.subplots_adjust(wspace=0.1, hspace=0.2)
        fig.tight_layout()
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "frame_{}.png".format("denoising")), dpi=1024)
        plt.show()

    def draw_figure_4_main(self, frame=0):
        plt.close("all")
        edges = ["edge 1", "edge 2", "edge 3"]
        box_colors = ["sandybrown", "limegreen", "cornflowerblue"]
        self._load_images(frame)

        self.image_process.load_image(self.images)
        self.image_process.align(preprocess=False)
        image = self.image_process.result().copy()

        # mask = self.image_process.get_mask(8)

        # fig, ax = plt.subplots(nrows=4, ncols=2, gridspec_kw={'height_ratios': [4, 1, 1, 1]})
        rcParams['axes.titlepad'] = 1
        rcParams['axes.titlesize'] = 8
        gs = GridSpec(4, 5, wspace=0.05, hspace=0.1)
        fig = plt.figure()
        plt.tight_layout()
        raw_img_trim = 35
        raw_ax = fig.add_subplot(gs[:, :2])
        # process_ax = [[fig.add_subplot(gs[i, j]) for i in np.arange(0, len(edges))] for j in np.arange(2, 5)]
        process_ax = [[fig.add_subplot(gs[:2, i]) for i in np.arange(2, 5)],
                      [fig.add_subplot(gs[2, i]) for i in np.arange(2, 5)],
                      [fig.add_subplot(gs[3, i]) for i in np.arange(2, 5)]]
        sizes_of_scalebars = [0.015, 0.03, 0.06, 0.06]
        img = image[raw_img_trim:-raw_img_trim, raw_img_trim:-raw_img_trim]
        raw_ax.imshow(img, cmap="gray")
        _, _, w, h = raw_ax.get_position().bounds
        raw_ax.add_artist(ScaleBar(pixel, "nm", length_fraction=0.3, width_fraction=sizes_of_scalebars[0],
                                   location="lower left", color="goldenrod", frameon=False,
                                   font_properties={"size": 12}))
        # ax[0].axis("equal")
        raw_ax.axis("off")
        # raw_ax.title.set_text("Raw Image")
        resize_to_width = None
        for i, (edge, box_color) in enumerate(zip(edges, box_colors)):
            mask = self.load_mask(edge)
            new_mask = self.image_process.cut_to_mask(mask)
            rect = self._get_rectangle(cv2.UMat(mask.copy().astype(np.uint8)), rotated=False)
            rect = tuple((tuple(np.array(rect[0]) - raw_img_trim), rect[1], rect[2]))
            self._draw_rectangle(rect, line_thickness=1, color=box_color, ax=raw_ax)

            self.image_process.normalize()
            # current_width = self.image_process.result().shape[1]
            # if resize_to_width is None:
            #     resize_to_width = current_width
            # resize_factor = resize_to_width/current_width
            # new_mask = self.image_process.upscale(resize_factor, mask=new_mask)
            img = self.image_process.result().copy()

            process_ax[i][0].imshow(img[2:-2, 2:-2, ], cmap="gray")
            _, _, w, h = process_ax[i][0].get_position().bounds
            # process_ax[i][0].title.set_text(self.image_process.titles[-1])
            process_ax[i][0].add_artist(ScaleBar(pixel, "nm",
                                                 length_fraction=0.5, width_fraction=sizes_of_scalebars[i + 1],
                                                 location="lower left", color="goldenrod", frameon=False,
                                                 font_properties={"size": 10}))
            process_ax[i][0].axis("off")
            self.image_process.denoise_bilateral()
            process_ax[i][1].imshow(self.image_process.result()[2:-2, 2:-2, ], cmap="gray")
            # process_ax[i][1].title.set_text(self.image_process.titles[-1])
            process_ax[i][1].axis("off")
            self.image_process.canny_devernay_edges(sigma=CANNY_SIGMA, mask=new_mask)
            # self.image_process.clean_up(5)
            # process_ax[i][2].title.set_text(self.image_process.titles[-2])
            process_ax[i][2].imshow(self.image_process.result()[2:-2, 2:-2, ], cmap="gray")
            process_ax[i][2].axis("off")
            self.image_process.revert(4)

        path = os.path.join(self.res_path, "for_suppl")
        fig.subplots_adjust(top=0.65)
        # fig.tight_layout()
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "frame_{}.png".format("detection")), dpi=2048)
        plt.show()
        plt.close()

    def average(self, images=None, n_images=1, sliding=False):
        if images is None:
            images = self.images.copy()

        images = np.array(images)
        if sliding:
            if n_images == 1:
                factors = np.array([0.75, 0.25])
            elif n_images == 2:
                factors = [0.75, 0.16, 0.08]
            else:
                factors = [0.75] + [0.12 / n for n in range(1, n_images + 1)]
            return np.sum(np.array([factor * images.copy()[n:-(n_images - n + 1)] / np.sum(factors)
                                    for factor, n in zip(factors, range(n_images, -1, -1))]),
                          axis=0)

        else:
            new_frame_number = images.shape[0] // n_images
            images = images[:new_frame_number * n_images]
            return np.mean(images.reshape(n_images, new_frame_number, images.shape[1], images.shape[2]), axis=0)
            # return block_reduce(images, block_size=(n_images, images.shape[1], images.shape[2]),
            #                     func=np.mean, func_kwargs={"axis": 0})

    def full_process_image(self):
        shape = self.image_process.result().shape
        image = resize(self.image_process.result(), (4 * shape[0], 4 * shape[1]))
        self.image_process.denoise_boxcar(kernel_radius=3)
        image = np.concatenate((image, resize(self.image_process.result(), (4 * shape[0], 4 * shape[1]))),
                               axis=1)
        self.image_process.revert(1)
        self.image_process.denoise_gaussian(3)
        image = np.concatenate((image, resize(self.image_process.result(), (4 * shape[0], 4 * shape[1]))),
                               axis=1)
        self.image_process.revert(1)
        self.image_process.denoise_nlm(fast=False)
        return np.concatenate((image, resize(self.image_process.result(), (4 * shape[0], 4 * shape[1]))),
                              axis=1)

    @staticmethod
    def _add_overlay(image, overlay_names: List[str], overlay_values: List[tuple]):
        image = np.stack((image,) * 3, axis=-1)
        dy = 20
        text_offset = 20
        for name, (value, units) in zip(overlay_names, overlay_values):
            overlay_line = "{}: {} {}".format(name, value, units)
            text_offset = text_offset + dy
            image = cv2.putText(image, overlay_line,
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                org=(10, text_offset),
                                fontScale=0.5,
                                color=(255, 0, 0),
                                lineType=2)
        return image

    def fluctuations_video(self, first_n_images=200, fps: int = None, FOV: str = None, T: int = None):
        self._load_images(frame=None, n_images=first_n_images)
        # mask = self.load_mask()

        # _ = self.image_process.cut_to_mask(mask)
        # self.image_process.denoise_nlm()
        name = "fluctuations_video_2"
        # images = self.average(n_images=3, sliding=False)
        images = self.average(n_images=1, sliding=True)
        self.image_process.load_image(images[0])
        self.image_process.align(preprocess=False)
        with tifffile.TiffWriter(os.path.join(self.res_path, name + '.tif')) as stack:
            for i, raw in enumerate(tqdm(images)):
                self.image_process.load_image(raw, label=i)
                self.image_process.align(preprocess=True)
                # smol_mask = self.image_process.cut_to_mask(mask)
                self.image_process.normalize()
                stack.write(img_as_ubyte(self.image_process.result(uint16=True)),
                            contiguous=False)

        print("Video *{}* was produced".format(name))

    def fluctuations_video_overlay(self, first_n_images=200):
        fps = 15
        FOV = 6
        T = 540
        self._load_images(frame=None, n_images=first_n_images)
        mask = self.load_mask()

        # _ = self.image_process.cut_to_mask(mask)
        # self.image_process.denoise_nlm()
        name = "fluctuations_video_2"
        # images = self.average(n_images=3, sliding=False)
        images = self.average(n_images=1, sliding=True)
        self.image_process.load_image(images[0])
        self.image_process.align(preprocess=False)
        time = np.round(np.arange(0, fps * images.shape[0], 1. / fps), 3)
        with tifffile.TiffWriter(os.path.join(self.res_path, name + '.tif')) as stack:
            for t, raw in tqdm(zip(time, images)):
                self.image_process.load_image(raw)
                self.image_process.align(preprocess=True)
                smol_mask = self.image_process.cut_to_mask(mask)
                self.image_process.normalize()
                self.image_process.upscale(4)
                image = self._add_overlay(img_as_ubyte(self.image_process.result(uint16=True)),
                                          ["time", "FOV", "Temp"],
                                          [(t, "s"), (FOV, r"micron"), (T, r"C")])
                stack.write(image,
                            contiguous=False)

            print("Video *{}* was produced".format(name))

    def draw_hannon_denoising(self, frame=0):
        plt.close("all")
        self._load_images(frame)
        self.get_ROI()
        self.image_process.global_hist_equal()
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].imshow(self.image_process.result(), cmap="gray")
        ax[0].axis("equal")
        ax[0].axis("off")
        ax[0].title.set_text("Original")
        self.image_process.denoise_boxcar(1)
        ax[1].imshow(self.image_process.result(), cmap="gray")
        ax[1].axis("equal")
        ax[1].axis("off")
        ax[1].title.set_text(self.image_process.titles[-1])
        plt.show()
        plt.close()


def _test_remove_background():
    process = ImageProcessor()
    bc_remove = BackgroundRemover()
    reader = RawReader()
    image = reader.read_single(os.path.join(SRC_FOLDER, REGION)).data
    image_nobc = bc_remove.remove(image)
    process.load_image(image)
    process.global_hist_equal()
    process.figure_result()
    process.load_image(image_nobc)
    process.global_hist_equal()
    process.figure_result()


class AnalysisFigureCreator(FigureCreator):
    def __init__(self, region, edge):
        super(AnalysisFigureCreator, self).__init__(region, edge)
        self.fft_analysis = FFTResults(self.res_path, original=True)
        self.distribution_analysis = DistributionResults(self.res_path, original=True)

    def draw_fft_dependency(self):
        # doing analysis manually to plot stuff
        n_all, q, y_q_msa = self.fft_analysis.get_y_q_msa()
        array_of_regions = np.array([4, 10, 24, 40, 50, -1])
        array_of_dependencies = np.array([2, 3, 4, 6, 8])
        n_all = np.arange(1, len(y_q_msa))
        one_over_q_all = (1 / q) ** 2
        fig, ax = plt.subplots(nrows=3, ncols=2)
        ax[0, 0].scatter(one_over_q_all, y_q_msa[1:], facecolors='none', edgecolors='k', marker='o',
                         alpha=0.8, s=8,
                         label="All data, $y_q vs 1/q$")
        ax[0, 0].legend()
        for i, (lower_lim, higher_lim, dependency) in enumerate(zip(array_of_regions[:-1],
                                                                    array_of_regions[1:],
                                                                    array_of_dependencies)):
            n = n_all[lower_lim:higher_lim]

            column = int((i + 1) / 3)
            row = int((i + 1) % 3)
            # y_q_msa = fft_calc._apply_gauss_factor(y_q_msa)
            real_life_1_over_q = self.fft_analysis.length / (2 * np.pi * n)

            proportional_q = real_life_1_over_q ** dependency

            slope, intercept, _, _, _ = stats.linregress(proportional_q, y_q_msa[n])
            print("slope = {} \n intercept = {}".format(round(slope, 5), round(intercept, 5)))
            ax[row, column].plot(proportional_q, (slope * proportional_q + intercept), color='b',
                                 linewidth=1.5, linestyle="--",
                                 label=r"$y_q \propto 1/q^{}$".format(dependency))
            selected_y = y_q_msa[n]
            ax[row, column].scatter(proportional_q, selected_y, facecolors='none', edgecolors='k', marker='o',
                                    alpha=0.8, s=8,
                                    label=r"Data for $\lambda \in$[{}, {}] nm".format(
                                        np.round(real_life_1_over_q[0] * 2 * np.pi, 1),
                                        np.round(real_life_1_over_q[-1] * 2 * np.pi, 1)))
            ax[row, column].legend()

        path = os.path.join(self.res_path, "for_suppl")
        # plt.subplots_adjust(wspace=0.1, hspace=0.2)
        fig.tight_layout()
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, "fft_comparisons.png"), dpi=1024)

        plt.show()
        plt.close()


def analysis_vs_region():
    the_colors = cm.get_cmap("rainbow")
    colors = the_colors(np.linspace(0, 1, len(_all_regions())))
    for i, region in enumerate(_all_regions()):
        for edge in _all_edges(region):
            res_path = os.path.join(RESULTS_FOLDER, region, edge)
            analyzer = DistributionResults(res_path, original=True)
            x, n, gauss_x, gauss, mean, std = analyzer.get_distribution()
            plt.plot(gauss_x, gauss, c=colors[i], alpha=0.8, label=region)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    plt.close()
    for i, region in enumerate(_all_regions()):
        for edge in _all_edges(region):
            res_path = os.path.join(RESULTS_FOLDER, region, edge)
            analyzer = FFTResults(res_path, original=True)
            n, q, y_q_msa = analyzer.get_y_q_msa()
            proportional_q = (1. / q) ** 2
            selected_y = y_q_msa[n]
            plt.plot(proportional_q, selected_y, c=colors[i], marker='o', alpha=0.8, label=region)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    plt.close()


def analysis_vs_integration_time():
    fps = [15, 20]
    regions = ['region 3, 15fps, try 12', 'region 3, 20fps, try 13']
    edges = ["edge 10", "edge 11"]
    for integration_time, region, edge in zip(fps, regions, edges):
        res_path = os.path.join(RESULTS_FOLDER, region, edge)
        analyzer = DistributionResults(res_path, original=True)
        x, n, gauss_x, gauss, mean, std = analyzer.get_distribution()
        plt.plot(gauss_x, gauss, alpha=0.8, label="FPS: {}".format(integration_time))
        plt.scatter(x, n, edgecolors='k', marker='o', alpha=0.8)
    plt.legend()
    plt.show()
    plt.close()
    for integration_time, region, edge in zip(fps, regions, edges):
        res_path = os.path.join(RESULTS_FOLDER, region, edge)
        analyzer = FFTResults(res_path, original=True)
        n, q, y_q_msa = analyzer.get_y_q_msa()
        proportional_q = (1. / q) ** 2
        selected_y = y_q_msa[n]
        plt.plot(proportional_q, selected_y, marker='o', alpha=0.8, label="FPS: {}".format(integration_time))
    plt.legend()
    plt.show()
    plt.close()


def analysis_vs_noise():
    noise_est = []
    processor = ImageProcessor()
    reader = RawReader()
    for region in _all_regions():
        src_path = os.path.join(SRC_FOLDER, region)
        list_of_images = reader.read_single(src_path, frames=(50, 100))
        noise_region = []
        for image in list_of_images:
            processor.load_image(image)
            noise_region.append(processor.estimate_noise())
        noise_est.append(np.average(np.array(noise_region)))
    the_colors = cm.get_cmap("rainbow")
    curvatures = np.array(noise_est)
    normalized_curvatures = (curvatures - np.min(curvatures)) / curvatures.ptp()
    color_array = the_colors(normalized_curvatures)
    for i, region in enumerate(_all_regions()):
        for edge in _all_edges(region):
            res_path = os.path.join(RESULTS_FOLDER, region, edge)
            analyzer = DistributionResults(res_path, original=True)
            x, n, gauss_x, gauss, mean, std = analyzer.get_distribution()
            plt.plot(gauss_x, gauss, c=color_array[i], alpha=0.8)
    plt.colorbar(cm.ScalarMappable(cmap=the_colors))
    plt.show()
    plt.close()
    for i, region in enumerate(_all_regions()):
        for edge in _all_edges(region):
            res_path = os.path.join(RESULTS_FOLDER, region, edge)
            analyzer = FFTResults(res_path, original=True)
            n, q, y_q_msa = analyzer.get_y_q_msa()
            proportional_q = (1. / q) ** 2
            selected_y = y_q_msa[n]
            plt.plot(proportional_q, selected_y, c=color_array[i], marker='o', alpha=0.8)
    plt.colorbar(cm.ScalarMappable(cmap=the_colors))
    plt.show()
    plt.close()


def analysis_vs_length():
    region = REGION
    edge = "edge 12"
    length_frac = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    res_path = os.path.join(RESULTS_FOLDER, region, edge)
    picklepath = os.path.join(res_path, "offsets_original.pickle")
    with open(picklepath, "rb") as f:
        offsets, lengths = pickle.load(f)
    length = np.average(lengths) * pixel
    fig, ax = plt.subplots(2, 1)
    for frac in length_frac:
        if frac == 1:
            shortened_offsets = offsets
        else:
            shortened_offsets = offsets[
                                int((1 - frac) * offsets.shape[0] / 2):-int((1 - frac) * offsets.shape[0] / 2), ]
        shortened_length = frac * length
        distribution_analyzer = ExperimentalDistributionResults(shortened_offsets, shortened_length)
        fft_analyser = ExperimentalFFTResults(shortened_offsets, shortened_length)
        x, n, gauss_x, gauss, mean, std = distribution_analyzer.get_distribution()
        ax[1].scatter(x, n, alpha=0.4, label="length fraction: {}".format(frac))
        ax[1].plot(gauss_x, gauss, alpha=0.8, label="length fraction: {}".format(frac))
        ax[1].set_xlim(-15, 15)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[1].legend(by_label.values(), by_label.keys())
        ax[1].set_aspect('auto')
        ax[1].set(xlabel=AxisNames.distr()["x"], ylabel=AxisNames.distr()["y"])
        n, q, y_q_msa = fft_analyser.get_y_q_msa()
        n = n[1:20]
        proportional_q = (1. / q[n]) ** 2
        ax[0].plot(proportional_q, y_q_msa[n], marker='o', alpha=0.8, label="length fraction: {}".format(frac))
        ax[0].legend()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel=AxisNames.fft()["x"], ylabel=AxisNames.fft()["y"])
    plt.tight_layout()
    plt.show()


def analysis_vs_curvature():
    curvatures = []
    successful_regions = []
    successful_edges = []
    edge_analyzer = EdgeAnalyzer()
    for region in _all_regions():
        for edge in _all_edges(region):
            res_path = os.path.join(RESULTS_FOLDER, region, edge)
            perpspath = os.path.join(res_path, "perpendiculars.pickle")
            with open(perpspath, "rb") as f:
                perps = pickle.load(f)
            edge_analyzer.perpendiculars_to_load(perp_arr=perps)
            mid_points = np.array([perpendicular.mid for perpendicular in edge_analyzer.perpendiculars])
            mid_points = measure.approximate_polygon(mid_points, 0.2)
            if np.sum(mid_points[1:, 0] >= mid_points[:-1, 0]) < 0.9 * mid_points.shape[0] and \
                    np.sum(mid_points[1:, 0] <= mid_points[:-1, 0]) < 0.9 * mid_points.shape[0]:
                mid_points = np.flip(mid_points, axis=1)
                if not np.sum(mid_points[1:, 0] >= mid_points[:-1, 0]) > 0.9 * mid_points.shape[0] or \
                        not np.sum(mid_points[1:, 0] <= mid_points[:-1, 0]) > 0.9 * mid_points.shape[0]:
                    print("{} - {} failed".format(region, edge))
                    continue
            mid_points = np.sort(mid_points, axis=0)
            curvature_spline = UnivariateSpline(mid_points[:, 0], mid_points[:, 1], k=2, s=12)
            curvature = np.average(np.abs(curvature_spline.derivative(2)(mid_points[:, 0])))
            if curvature < 0.1:
                curvatures.append(curvature)
                successful_regions.append(region)
                successful_edges.append(edge)
            # fig, ax = plt.subplots(2, 1)
            # ax[0].plot(mid_points[:, 0], mid_points[:, 1])
            # ax[0].plot(mid_points[:, 0], curvature_spline(mid_points[:, 0]))
            # ax[0].set_aspect('equal')
            # ax[1].plot(mid_points[:, 0], np.abs(curvature(mid_points[:, 0])))

            # plt.show()
    the_colors = cm.get_cmap("rainbow")
    curvatures = np.array(curvatures)
    normalized_curvatures = (curvatures - np.min(curvatures)) / curvatures.ptp()
    color_array = the_colors(normalized_curvatures)
    for i, (edge, region) in enumerate(zip(successful_edges, successful_regions)):
        res_path = os.path.join(RESULTS_FOLDER, region, edge)
        analyzer = DistributionResults(res_path, original=True)
        x, n, gauss_x, gauss, mean, std = analyzer.get_distribution()
        plt.plot(gauss_x, gauss, c=color_array[i], alpha=0.8)
    plt.colorbar(cm.ScalarMappable(cmap=the_colors))
    plt.show()
    plt.close()
    for i, (edge, region) in enumerate(zip(successful_edges, successful_regions)):
        res_path = os.path.join(RESULTS_FOLDER, region, edge)
        analyzer = FFTResults(res_path, original=True)
        n, q, y_q_msa = analyzer.get_y_q_msa()
        n = n[1:20]
        proportional_q = (1. / q[n]) ** 2
        selected_y = y_q_msa[n]
        plt.plot(proportional_q, selected_y, c=color_array[i], marker='o', alpha=0.8)
    plt.colorbar(cm.ScalarMappable(cmap=the_colors))
    plt.show()
    plt.close()


def analysis_vs_averaging():
    region = REGION
    averages = [1, 2, 5, 10]
    imgs, mask = load_experimental_images(os.path.join(SRC_FOLDER, region))
    ax, fig = plt.subplots(1, 2)
    for average in averages:
        # Get offsets
        coords = experimental_manual_masking(imgs.copy(), mask.copy(), align=True, average=average)
        coords_to_offsets = EdgeAnalyzer(coords)
        coords_to_offsets.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=32)
        # coords_to_offsets.show_perps()
        positions, _ = coords_to_offsets.get_edge_variations(savefig=None)
        edge_length = coords_to_offsets.get_length()
        # Get distribution
        offsets_to_distr = ExperimentalDistributionResults(positions, edge_length)
        x, n, gauss_x, gauss, mean, std = offsets_to_distr.get_distribution()
        ax[1].plot(gauss_x, gauss, alpha=0.8, label="averaging: {} images".format(average))
        ax[1].legend()
        ax[1].set(xlabel=AxisNames.distr()["x"], ylabel=AxisNames.distr()["y"])
        # Get FFT
        offsets_to_fft = ExperimentalFFTResults(positions, edge_length)
        n, q, y_q_msa = offsets_to_fft.get_y_q_msa()
        proportional_q = (1. / q) ** 2
        ax[0].plot(proportional_q, y_q_msa, marker='o', alpha=0.8, label="averaging: {} images".format(average))
        ax[0].set(xlabel=AxisNames.fft()["x"], ylabel=AxisNames.fft()["y"])
        ax[0].legend()
    plt.show()


def analysis_vs_experiment_duration():
    region = REGION
    edge = "edge 1"
    length_frac = [1, 0.5, 0.25, 0.1, 0.01, 0.005]
    res_path = os.path.join(RESULTS_FOLDER, region, edge)
    picklepath = os.path.join(res_path, "offsets_original.pickle")
    with open(picklepath, "rb") as f:
        offsets, lengths = pickle.load(f)
    length = np.average(lengths)
    fig, ax = plt.subplots(1, 2)
    for frac in length_frac:
        if frac == 1:
            shortened_offsets = offsets
        else:
            shortened_offsets = offsets[:,
                                int((1 - frac) * offsets.shape[1] / 2):-int((1 - frac) * offsets.shape[1] / 2), ]
        distribution_analyzer = ExperimentalDistributionResults(shortened_offsets, length)
        fft_analyser = ExperimentalFFTResults(shortened_offsets, length)
        x, n, gauss_x, gauss, mean, std = distribution_analyzer.get_distribution()
        ax[1].scatter(x, n, alpha=0.4, label="duration fraction: {}".format(frac))
        ax[1].plot(gauss_x, gauss, alpha=0.8, label="duration fraction: {}".format(frac))
        # ax[1].set_xlim(-15, 15)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[1].legend(by_label.values(), by_label.keys())
        ax[1].set_aspect('auto')
        ax[1].set(xlabel=AxisNames.distr()["x"], ylabel=AxisNames.distr()["y"])
        n, q, y_q_msa = fft_analyser.get_y_q_msa()
        y_q_msa = y_q_msa
        ax[0].plot(1 / (q ** 2), y_q_msa, marker='o', alpha=0.8, label="duration fraction: {}".format(frac))
        ax[0].legend()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel=AxisNames.fft()["x"], ylabel=AxisNames.fft()["y"])
    plt.tight_layout()
    plt.show()


def analysis_vs_cap_in_amplitude():
    caps = [8, 10, 20, 100]
    region = REGION
    edge = "edge 12"
    res_path = os.path.join(RESULTS_FOLDER, region, edge)
    picklepath = os.path.join(res_path, "offsets_original.pickle")
    with open(picklepath, "rb") as f:
        offsets, lengths = pickle.load(f)
    length = np.average(lengths) * pixel
    offsets = offsets * pixel

    fig, ax = plt.subplots(2, 1)
    for cap in caps:
        capped_offsets = offsets.copy()
        capped_offsets[capped_offsets > cap] = cap
        capped_offsets[capped_offsets < -cap] = -cap
        distribution_analyzer = ExperimentalDistributionResults(capped_offsets, length)
        fft_analyser = ExperimentalFFTResults(capped_offsets, length)
        x, n, gauss_x, gauss, mean, std = distribution_analyzer.get_distribution()
        ax[1].scatter(x, n, alpha=0.4, label="Max offset: {}nm".format(cap))
        ax[1].plot(gauss_x, gauss, alpha=0.8, label="Max offset: {}nm".format(cap))
        ax[1].set_xlim(-15, 15)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[1].legend(by_label.values(), by_label.keys())
        ax[1].set_aspect('auto')
        ax[1].set(xlabel=AxisNames.distr()["x"], ylabel=AxisNames.distr()["y"])
        n, q, y_q_msa = fft_analyser.get_y_q_msa()
        n = n[1:20]
        proportional_q = (1. / q[n]) ** 2
        ax[0].plot(proportional_q, y_q_msa[n], marker='o', alpha=0.8, label="Max offset: {}nm".format(cap))
        ax[0].legend()
        ax[0].set_aspect('auto')
        ax[0].set(xlabel=AxisNames.fft()["x"], ylabel=AxisNames.fft()["y"])
    plt.tight_layout()
    plt.show()


def analysis_vs_position_in_frame():
    region = REGION
    source_folder = os.path.join(SRC_FOLDER, region)
    # set up plot
    gs = GridSpec(2, 3, wspace=0.05, hspace=0.1)
    fig = plt.figure(figsize=(4, 6))
    ax_image = fig.add_subplot(gs[:, :2])
    ax_fft = fig.add_subplot(gs[0, 2])
    ax_distr = fig.add_subplot(gs[1, 2])

    read = RawReader()
    # get image to show edges on
    processor = ImageProcessor()

    image = read.read_single(source_folder, frames=0).data
    processor.load_image(image)
    processor.clahe_hist_equal()
    processor.denoise_nlm(fast=False)
    image = np.stack((processor.result(),) * 3, axis=-1)
    ax_image.imshow(image)
    colors = ["firebrick", "orange", "palegreen", "dodgerblue", "orchid", "moccasin"]
    for i, edge in enumerate(_all_edges(region)):
        res_path = os.path.join(RESULTS_FOLDER, region, edge)
        edge_coords = load_pickle(os.path.join(res_path, 'coordinates_canny.pickle'))[0]
        ax_image.scatter(edge_coords[:, 1], edge_coords[:, 0], color=colors[i], alpha=0.8)
        analyzer = DistributionResults(res_path, original=True)
        x, n, gauss_x, gauss, mean, std = analyzer.get_distribution()
        ax_distr.plot(gauss_x, gauss, c=colors[i], alpha=0.8)

        analyzer = FFTResults(res_path, original=True)
        n, q, y_q_msa = analyzer.get_y_q_msa()
        proportional_q = (1. / q) ** 2
        selected_y = y_q_msa[n]
        ax_fft.plot(proportional_q, selected_y, c=colors[i], marker='o', alpha=0.8)
    ax_distr.set_xlim(-15, 15)
    plt.show()


def draw_edges_with_time():
    region = REGION
    edge = "edge 12"
    fps = 20
    res_path = os.path.join(RESULTS_FOLDER, region, edge)
    picklepath = os.path.join(res_path, "offsets_original.pickle")
    with open(picklepath, "rb") as f:
        positions, lengths = pickle.load(f)
    start_frame = 10
    N_frames = 5
    delta_frames = 5
    fig, ax = plt.subplots(N_frames, 1)
    x_axis = np.linspace(0, np.average(lengths), positions.shape[0])
    positions = pixel * positions
    for i in range(N_frames):
        frame = start_frame + i * delta_frames
        time_difference = "{} s".format(np.round((1. / fps) * i * delta_frames, 5))
        ax[i].plot(x_axis, positions[:, frame], label=time_difference)
        ax[i].set_ylim(-20, 20)
        ax[i].legend()
        if i < N_frames - 1:
            ax[i].get_xaxis().set_ticks([])
    plt.show()


def show_detection():
    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.clahe_hist_equal()
    mask = processor.get_mask(10)
    mask = processor.cut_to_mask(mask)
    # processor.denoise_nlm(fast=True)
    #
    # processor.canny_edges(sigma=1, mask=None)
    # processor.clean_up(5)
    # processor.plot_all()
    processor.figure_result()


def get_analysis_figure(region, edge):
    results_path = os.path.join(RESULTS_FOLDER, region, edge)
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    distribution_analysis(results_path, adjusted=False, ax=ax[1])
    fft_analysis(results_path, adjusted=False, ax=ax[0])
    plt.savefig(os.path.join(results_path, 'double_fig.png'), dpi=1000, transparent=False)


if __name__ == '__main__':
    # for i, path in enumerate(video_paths):
    #     raw_to_video(path, destination_path=target_folder, fps=20, name="region" + str(i+10))
    # _test_remove_background()
    region = "run 6"
    edge = "edge 1"
    fig = FigureCreator(region, edge)
    # fig.draw_figure_4_main()
    # get_analysis_figure(region, edge)
    # fig.draw_figure_1()
    # fig.draw_figure_2()
    # fig.draw_figure_3()
    # fig.draw_figure_4()
    fig.fluctuations_video_overlay(300)
    # analysis_vs_cap_in_amplitude()
    # analysis_vs_experiment_duration()
    # show_detection()
