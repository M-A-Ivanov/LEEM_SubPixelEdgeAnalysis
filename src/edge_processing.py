import os
from typing import List
from tqdm import tqdm
import numpy as np
from skimage import morphology, measure
from scipy import stats, spatial

import matplotlib.pyplot as plt

import seaborn as sns

from data_analysis import FFTResults
from global_paths import SRC_FOLDER
from image_processing import ImageProcessor
from perpendiculars import PerpendicularAdjuster, PerpendicularProj
from raw_reader import RawReader

# sns.set_context('paper')  # 'talk', 'paper', 'poster'
pixel = 3000 / 1024  # nm


class EdgeAnalyzer:
    def __init__(self, edges=None, low_memory=True):
        if edges is not None:
            self.edges: List = [np.flip(edge, axis=1) for edge in edges]
        else:
            self.edges = None
        self.edges_lengths = []
        self.order_edges()
        self.perpendiculars: List[PerpendicularAdjuster] = []
        self.perp_args = None
        self.low_memory = low_memory

    def get(self):
        return self.edges

    def get_length(self):
        return np.mean(np.squeeze(np.array(self.edges_lengths)))

    def order_edges(self):
        if self.edges is None:
            return

        def get_closest_point(point, other_points):
            # dist = np.sum((other_points - point) ** 2, axis=1)
            dist = spatial.distance.cdist(other_points, point.reshape(1, 2))
            return np.argmin(dist), np.min(dist)

        def plug_point(point, other_points):
            dist = spatial.distance.cdist(other_points, point.reshape(1, 2))
            pair_sums = dist[:-1] + dist[1:]
            other_points = np.insert(other_points, np.argmin(pair_sums) + 1, point, axis=0)[:-1]
            return other_points

        threshhold_dist = 10
        for i, edge in enumerate(self.edges):
            save_length = len(edge)
            real_length = 0
            if save_length == 0:
                print("Length of edge {} was 0.".format(i))
            else:
                start_from = 0
                new_order = np.zeros_like(edge)
                new_order[0] = edge[start_from]
                edge = np.delete(edge, start_from, axis=0)
                for j in range(len(edge)):
                    idx, distance = get_closest_point(new_order[j], edge)
                    if distance > threshhold_dist:
                        new_order[:j + 1] = new_order[:j + 1][::-1]
                        idx, distance = get_closest_point(new_order[j], edge)
                        if distance > threshhold_dist:
                            new_order = plug_point(edge[idx], new_order)
                            edge = np.delete(edge, idx, axis=0)
                            continue
                    real_length += distance
                    new_order[j + 1] = edge[idx]

                    edge = np.delete(edge, idx, axis=0)
                self.edges[i] = new_order
                self.edges_lengths.append(real_length)
                # self._plot_edge(new_order)
                assert len(self.edges[i]) == save_length

    def _plot_edge(self, edge):
        plt.figure()
        plt.scatter(edge[:, 1], edge[:, 0])
        plt.plot(edge[:, 1], edge[:, 0])
        plt.axis("equal")
        plt.show()
        plt.close()

    @staticmethod
    def _load_images(region, frames):
        reader = RawReader()
        if frames is None:
            list_of_images = [im.data for im in reader.read_folder(os.path.join(SRC_FOLDER, region))]
        else:
            list_of_images = reader.read_single(folder_path=os.path.join(SRC_FOLDER, region), frames=frames).data
        return list_of_images

    def perpendiculars_to_save(self):
        perpoints = np.zeros((len(self.perpendiculars), 2, 2))
        for i, perp in enumerate(self.perpendiculars):
            perpoints[i, 0, ] = perp.mid
            perpoints[i, 1, ] = perp.second
        return perpoints

    def perpendiculars_to_load(self, perp_arr, low_memory=True):
        self.perpendiculars = [PerpendicularAdjuster(arr[0], arr[1]) for arr in perp_arr]
        for perp in self.perpendiculars:
            perp.low_memory = low_memory

    def get_perpendiculars_simple(self, pixel_average=10, edge_frame=None, savefig=None):
        """ Simpler creation of perpendiculars. Makes them with respect to the average direction of
        the edge. """
        if edge_frame is None:
            edge_frame = 0
        edge = self.edges[edge_frame]
        # TODO: if y range bigger than x, swap axes
        # average direction
        slope, intercept, r, p, se = stats.linregress(edge[:, 0], edge[:, 1])
        if np.isclose(slope, 0):
            for edge_point in edge:
                self.perpendiculars.append(PerpendicularAdjuster(mid_pt=edge_point,
                                                                 second_pt=(edge_point + np.array([1, 0]))))
            return

        perp_slope = -1./slope
        relevant_x_coordinates = np.arange(np.ceil(np.min(edge[:, 0]))+pixel_average,
                                           np.floor(np.max(edge[:, 0]))-pixel_average)
        for x in relevant_x_coordinates:
            mid = np.array([x, slope * x + intercept])
            second = np.array([x + 1, perp_slope * (x + 1) + (slope - perp_slope)*x + intercept])
            self.perpendiculars.append(PerpendicularAdjuster(mid_pt=mid, second_pt=second))
        if savefig is not None:
            plt.figure()
            plt.scatter(edge[:, 0], edge[:, 1], alpha=0.8)
            for perp in self.perpendiculars[::2]:
                plt.scatter(perp.mid[0], perp.mid[1], color='green')  # the midpoints of fit lines
                second_pt_extended = perp.get_point_at_dist(10)
                plt.scatter((perp.mid[0], second_pt_extended[0]),
                            (perp.mid[1], second_pt_extended[1]), color='red')
                plt.axline(perp.mid, perp.second, linestyle='dashed')  # the perpendiculars
            plt.axis("equal")
            plt.savefig(os.path.join(savefig, "perpendiculars_{}.png".format(edge_frame)))
            plt.close()
        else:
            plt.show()

    def get_perpendiculars_adaptive(self, every_point=True, edge_frame=None, pixel_average=8, savefig=None):
        """
        Adaptive perpendiculars creation. They follow the approximate local shape of the step.

        Args:
            persistent: bool;   whether to keep the first calculated perpendiculars throughout.
            edge_frame: int;    choose which video frame we use for the calculation of the perpendiculars.
                        TODO: if None, get perpendiculars for all edges at the same time? otherwise, make default = 0

            pixel_average: int; perpendiculars are the bisection (symmetral) of the line fit of every *pixel_average*
                                number of points on the chosen edge.
            savefig: str;       path to save the resulting plot of the edge with the perpendiculars.
                                If None, no plot is saved.

        Returns: Creates the perpendiculars inside the EdgeAnalyzer, returns nothing.

        """

        self.perp_args = [every_point, edge_frame, pixel_average, savefig]
        self.perpendiculars = []
        if edge_frame is None:
            edge_frame = 0
        edge = self.edges[edge_frame]
        truncation = len(edge) % pixel_average
        if truncation:
            edge = edge[int(truncation / 2):-(int(truncation / 2) + int(truncation % 2))]
        assert len(edge) % pixel_average == 0

        if savefig is not None:
            plt.figure()
            plt.scatter(edge[:, 0], edge[:, 1], alpha=0.8)

        def get_perp(segment):

            def point_on_line(a, b, p):
                ap = p - a
                ab = b - a
                if (ap[0] ** 2 + ap[1] ** 2) != 0 or (ab[0] ** 2 + ab[1] ** 2) != 0:
                    ret = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
                else:
                    ret = p
                return ret

            slope, intercept, r, p, se = stats.linregress(segment[:, 0], segment[:, 1])
            if np.isnan((slope, intercept)).any():
                pt1, pt2 = np.array([segment[0, 0], segment[0, 1]]), \
                           np.array([segment[-1, 0], segment[-1, 1]])
                mid_point = np.mean([pt1, pt2], axis=0)
            else:
                pt1, pt2 = np.array([segment[0, 0], slope * segment[0, 0] + intercept]), \
                           np.array([segment[-1, 0], slope * segment[-1, 0] + intercept])
                mid_point = np.mean(
                    [point_on_line(pt1, pt2, segment[0]), point_on_line(pt1, pt2, segment[-1])],
                    axis=0)
            perpt = np.array([mid_point[0] - pt2[1] + pt1[1], mid_point[1] + pt2[0] - pt1[0]])
            if not np.isnan((mid_point, perpt)).any():
                return PerpendicularAdjuster(mid_pt=mid_point, second_pt=perpt)
            else:
                return None

        print("Creating perpendiculars (bisectors)...")
        if every_point:
            for i in tqdm(range(int(pixel_average / 2), (len(edge) - int(pixel_average / 2))), leave=False):
                segment = measure.approximate_polygon(edge[i - int(pixel_average / 2):(i + int(pixel_average / 2))],
                                                      tolerance=2)

                perp = get_perp(segment)
                if perp is not None:
                    self.perpendiculars.append(perp)
                    if savefig is not None:
                        if i % 2 == 0:
                            # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', alpha=0.5) # the fitted lines
                            plt.scatter(perp.mid[0], perp.mid[1], color='green')  # the midpoints of fit lines
                            second_pt_extended = perp.get_point_at_dist(10)
                            plt.scatter((perp.mid[0], second_pt_extended[0]),
                                        (perp.mid[1], second_pt_extended[1]), color='red')
                            plt.axline(perp.mid, perp.second, linestyle='dashed')  # the perpendiculars
        else:
            for i in range(int(len(edge) / pixel_average)):
                segment = edge[i * pixel_average:(i + 1) * pixel_average]
                perp = get_perp(segment)
                perp.set_to_low_memory()
                if perp is not None:
                    self.perpendiculars.append(perp)

                    if savefig is not None:
                        # plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='red', alpha=0.5)      # the fitted lines
                        plt.scatter(perp.mid[0], perp.mid[1], color='green')  # the midpoints of fit lines
                        plt.scatter((perp.mid[0], perp.second[0]), (perp.mid[1], perp.second[1]), color='red')
                        plt.axline(perp.mid, perp.second, linestyle='dashed')  # the perpendiculars
        if savefig is not None:
            plt.axis("equal")
            plt.savefig(os.path.join(savefig, "perpendiculars.png"))
            plt.close()

    def show_perps(self, edge_frame=1, savefig=None):
        edge = self.edges[edge_frame]
        plt.figure(figsize=(12.8, 9.6), dpi=100)
        plt.plot(edge[:, 0], edge[:, 1], alpha=0.8)
        plt.axis("equal")
        for perp in self.perpendiculars:
            plt.scatter(perp.mid[0], perp.mid[1], color='green', alpha=0.5, s=1)  # the midpoints of fit lines
            plt.scatter(perp.mid[0], perp.mid[1], color='red', alpha=0.2, s=1.2)
            plt.axline(perp.mid, perp.second, linestyle='dashed', alpha=0.1)  # the perpendiculars
        if savefig is not None:
            plt.axis("equal")
            plt.savefig(os.path.join(savefig, "perpendiculars_{}.png".format(edge_frame)))
            plt.close()
        else:
            plt.show()

    def get_edge_variations(self, savefig=None, frames: list or tuple or int = None):
        positions = np.zeros((len(self.perpendiculars), len(self.edges)), dtype=float)
        interpolation_success_rate = np.zeros(2)
        if frames is not None:
            if isinstance(frames, tuple) or isinstance(frames, list):
                edges = self.edges[frames[0]:frames[1]]
            elif isinstance(frames, int):
                edges = [self.edges[frames]]
            else:
                print("frames argument is not valid.")
                return
        else:
            edges = self.edges
        print("Finding rough detections for each frame...")
        missing_edges = np.zeros(len(edges), dtype=bool)
        for j, edge in enumerate(tqdm(edges, leave=False)):
            if savefig is not None:
                self.show_perps(edge_frame=j)
            for i, perp in enumerate(self.perpendiculars):
                try:
                    positions[i, j] = perp.project_weno(edge)
                    interpolation_success_rate[0] += 1
                except Exception as e:
                    try:
                        positions[i, j] = perp.project_interpolation(edge)
                        interpolation_success_rate[1] += 1
                    except Exception as e:
                        try:
                            # print("Interpolation during detection failed with exception:" + str(e))
                            positions[i, j] = perp.project_simple(edge)
                            interpolation_success_rate[1] += 1
                        except Exception as e:
                            print("Assuming frame {} is incomplete. Removing".format(j))
                            missing_edges[j] = True
                            break

        positions = positions[:, ~missing_edges]
        info = {
            "interpolation rate": (100 * (interpolation_success_rate[0] / np.sum(interpolation_success_rate)), "%"),
            "total detections": (np.sum(interpolation_success_rate), None),
        }
        return positions, info

    def adjust_edge_variations(self, region_path: str, max_dist: int, rough_offsets=None,
                               adjustment_type='devernay', frames: list or tuple or int = None):

        list_of_images = self._load_images(region_path, frames)
        processor = ImageProcessor()
        mask_image = np.zeros_like(list_of_images[0], dtype=np.uint8)
        for perp in self.perpendiculars:
            mask_image[int(perp.mid[1]), int(perp.mid[0])] = 1
        selem = morphology.selem.disk(max_dist)
        mask_image = morphology.binary_dilation(mask_image, selem=selem)
        coordinates = np.ix_(mask_image.any(1), mask_image.any(0))
        cut_corrections = np.array([np.min(coordinates[1]), np.min(coordinates[0])])

        for perp in self.perpendiculars:
            perp.mid = np.subtract(perp.mid, cut_corrections)
            perp.second = np.subtract(perp.second, cut_corrections)

        def standard_preprocess(image):
            processor.load_image(image)
            processor.align(preprocess=True)
            _ = processor.cut_to_mask(mask_image)
            processor.global_hist_equal()
            processor.denoise_nlm(fast=False)
            # processor.denoise_boxcar(1)

        def reject_outliers(data, m=2.):
            d = np.abs(data - np.median(data))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            data[s >= m] = 0
            return data

        positions = np.zeros((len(self.perpendiculars), len(self.edges)), dtype=float)
        offset_corrections = np.zeros_like(positions, dtype=float)
        corrections_success_rate = np.zeros(2)
        print("Adjusting positions for each frame...")
        for j, edge in enumerate(tqdm(self.edges[1:])):
            standard_preprocess(list_of_images[j])

            for i, perp in enumerate(self.perpendiculars):
                if rough_offsets is not None:
                    detection = rough_offsets[i, j]
                else:
                    detection = 0
                try:
                    if adjustment_type == "tanh":
                        offset_corrections[i, j] = perp.adjust_tanh(processor.result(), radius=max_dist,
                                                                    rough=detection,
                                                                    preprocess_locally=False)
                    if adjustment_type == "devernay":
                        offset_corrections[i, j] = perp.adjust_devernay(processor.result(), radius=max_dist,
                                                                        rough=detection,
                                                                        preprocess_locally=False)
                    corrections_success_rate[0] += 1
                except RuntimeError:
                    corrections_success_rate[1] += 1
        offset_corrections = np.apply_along_axis(reject_outliers, axis=0, arr=offset_corrections)
        # successful_corrections = 100 * np.count_nonzero(offset_corrections) / offset_corrections.size
        info = {"tanh adjustment rate": (100 * (corrections_success_rate[0] / np.sum(corrections_success_rate)), "%"),
                "average adjustment": (np.average(offset_corrections[np.where(offset_corrections != 0)]), None)}

        print(info)
        return np.add(positions, offset_corrections), info

    def clear_perpendiculars(self):
        for perp in self.perpendiculars:
            perp.clear()

    @staticmethod
    def dilate_edge(edge_img):
        disk = morphology.disk(4)
        edge = morphology.dilation(edge_img, selem=disk)
        edge_img = morphology.skeletonize(edge)
        return edge_img
