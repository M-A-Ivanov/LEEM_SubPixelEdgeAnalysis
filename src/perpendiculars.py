from typing import List

import numpy as np
from skimage import transform, measure
from scipy import stats, optimize

from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import math

from scipy import interpolate
from weno4 import weno4

from image_processing import ImageProcessor


class Perpendicular:
    def __init__(self, mid_pt, second_pt):
        self.mid = mid_pt.astype(float)
        self.second = second_pt.astype(float)


class PerpendicularProj(Perpendicular):
    def __init__(self, mid_pt, second_pt):
        super(PerpendicularProj, self).__init__(mid_pt, second_pt)
        self.second = self.get_point_at_dist(1)
        self.threshold_to_perp = 5
        self.threshold_on_perp = 20
        self.found_offsets = []
        self.found_points = []
        self.low_memory = False

    def set_to_low_memory(self):
        self.low_memory = True

    def clear(self):
        self.found_points = []
        self.found_offsets = []

    def get_point_at_dist(self, dist, ):
        return self.mid + dist * self.get_perp_direction()

    def get_perp_direction(self):
        v = (self.second - self.mid)
        return v / np.linalg.norm(v)

    def angle_to_y(self):
        """Counterclockise angle in degrees"""
        normalized = self.get_perp_direction()
        y_axis = np.array([0, 1])
        return np.rad2deg(np.arctan2(np.cross(y_axis, normalized), np.dot(y_axis, normalized)))

    def point_on_line(self, p):
        ap = p.astype(float) - self.mid.astype(float)
        ab = self.second - self.mid
        if (ap[0] ** 2 + ap[1] ** 2) != 0 or (ab[0] ** 2 + ab[1] ** 2) != 0:
            ret = self.mid + np.dot(ap, ab) / np.dot(ab, ab) * ab
        else:
            ret = p
        return ret

    @staticmethod
    def dist_to_point(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def dist_to_line(self, p):
        pt_on_line = self.point_on_line(p)
        return self.dist_to_point(p, pt_on_line)

    def sign_yaxis(self, pt_on_perp):
        """ Which side of the x-axis the point is"""
        return np.sign(np.dot((pt_on_perp - self.mid), (self.second - self.mid)))

    def sign_xaxis(self, point):
        """ Which side of the y-axis the point is"""
        return np.sign(np.cross((point - self.mid), (self.second - self.mid)))

    def line_intersection(self, line2):
        xdiff = (self.mid[0] - self.second[0], line2[0][0] - line2[1][0])
        ydiff = (self.mid[1] - self.second[1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(self.mid, self.second), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div

        assert self._is_on_line((x, y))
        return tuple((x, y))

    def f(self, x):
        m = (self.second[1] - self.mid[1]) / (self.second[0] - self.mid[0])
        return self.second[0] + m * (x - self.mid[0])

    def _is_on_line(self, point):
        return abs(point[1] - self.f(point[0])) <= 1

    def to_local_transform(self, points_arr):
        # return np.array([self._affine_transform(points_arr[..., i]) for i in range(points_arr.shape[1])])
        return np.apply_along_axis(self._affine_transform, axis=0, arr=points_arr)

    def _affine_transform(self, point):
        pt_on_perp = self.point_on_line(point)
        new_y_coord = self.dist_to_point(pt_on_perp, self.mid)
        new_x_coord = self.dist_to_point(pt_on_perp, point)
        new_y_coord = self.sign_yaxis(pt_on_perp) * new_y_coord
        new_x_coord = new_x_coord * self.sign_xaxis(point)
        return np.array([new_x_coord, new_y_coord], dtype=float)

    def to_global_transform(self, points):
        """ Not working atm. TODO: create a simple test and fix. """
        y_ordinate = np.array([0, 1], dtype=float)
        perp_vector = self.get_perp_direction()
        # theta = np.arccos(np.dot(perp_vector, y_ordinate))
        theta = np.pi - np.arccos(self.get_perp_direction())
        RotMatrix = np.zeros((2, 2))
        RotMatrix[0][0] = np.cos(theta)
        RotMatrix[0][1] = -1 * np.sin(theta)
        RotMatrix[1][0] = np.sin(theta)
        RotMatrix[1][1] = np.cos(theta)
        # points = np.sum((points, self.mid), axis=1)
        return np.apply_along_axis(lambda arr: np.dot(RotMatrix, arr) + self.mid, axis=0, arr=points)

    def _find_n_closest(self, points, n_closest, threshold=10):
        if len(points) == n_closest:
            return points
        points = [point for point in points if self.dist_to_point(self.point_on_line(point), self.mid) < threshold]
        points = np.array(points)
        dists = np.array([self.dist_to_point(self.point_on_line(point), point) for point in points])
        # find_side = np.vectorize(self.sign_yaxis, excluded='self')
        closest_n_indices = np.argpartition(dists, n_closest)[:n_closest]
        # check_sides = np.unique(find_side(points[closest_n_indices]))
        # if len(check_sides) == 1:
        #     signs = find_side(points)
        #     other_side_dists = dists[np.where(signs != check_sides)]
        #     added_point = points[np.argmin(other_side_dists)]
        #     return np.append(points[closest_n_indices], added_point)
        # else:
        return points[closest_n_indices]

    def _to_local_coordinate_system(self, points, n_closest=None):
        if n_closest is not None:
            assert n_closest <= points.shape[0]
            points = self._find_n_closest(points, n_closest, threshold=10)
        if points.ndim == 1:  # if a single point is given
            points = np.expand_dims(points, axis=0)
        points_original = points.copy()
        points = points.astype(float)
        i = 0
        if points_original.ndim == 1:  # if a single point is given
            points_original = np.expand_dims(points_original, axis=0)
        for point in points_original:
            new_pt = self._affine_transform(point)
            if new_pt[0] in points[:i, 0]:
                idx = np.argwhere(new_pt[0] == points[:i, 0])
                points[idx, 1] = np.average((points[idx, 1], new_pt[1]))
                points = points[:-1]
            else:
                points[i] = new_pt
                i += 1
        del i
        return np.squeeze(points[points[:, 0].argsort()])

    def get_local_image(self, image, radius: int = 20, offset_to_mid=0, rotation=False):
        center = self.mid + offset_to_mid * self.get_perp_direction()
        center = center.astype(int)
        assert (center - radius >= 0).all()
        assert (center[1] + radius <= image.shape[0]).all()
        assert (center[0] + radius <= image.shape[1]).all()
        if rotation:
            rot_angle = self.angle_to_y()
            image = transform.rotate(image, rot_angle, center=(center[0], center[1]))
        extent = [- radius, radius, - radius, radius]
        return image[(center[1] + extent[0]):(center[1] + extent[1]),
               (center[0] + extent[2]):
               (center[0] + extent[3]), ], extent

    def project_simple(self, points: List or np.ndarray):
        distance_to_line = 1000
        offset = None
        original_point = None
        proj_point = None
        saved_sign = None
        for point in points:
            pt_on_perp = self.point_on_line(point)
            dist_to_mid = self.dist_to_point(pt_on_perp, self.mid)
            dist_to_perp = self.dist_to_point(pt_on_perp, point)
            sign = self.sign_yaxis(pt_on_perp)
            if dist_to_mid < self.threshold_on_perp:
                if dist_to_perp < distance_to_line:
                    offset = sign * dist_to_mid
                    original_point = point
                    proj_point = pt_on_perp
                    saved_sign = sign
                    distance_to_line = dist_to_perp
        if not self.low_memory:
            self.found_offsets.append(offset)
            local_original = self._to_local_coordinate_system(original_point)
            local_projection = self._to_local_coordinate_system(proj_point)
            detection_illustration = np.stack((np.linspace(local_original[0], local_projection[0], 1000),
                                               np.linspace(local_original[1], local_projection[1], 1000)))
            self.found_points.append([local_original, detection_illustration, local_projection, saved_sign])

        return offset

    def project_interpolation(self, points: List or np.ndarray, are_local=False):
        n_points = 16
        points = np.array(points, dtype=float)
        if not are_local:
            points = self._to_local_coordinate_system(points, n_closest=n_points)
        interp = self._linear_interpolate(points)
        # interp = interpolate.UnivariateSpline(points[:, 0], points[:, 1])
        offset = interp(0)
        interp_x_for_plot = np.linspace(min(points[:, 0]), max(points[:, 0]), 1000)
        interp_y_for_plot = interp(interp_x_for_plot)
        interpolation_for_plot = np.stack((interp_x_for_plot,
                                           interp_y_for_plot))
        if not self.low_memory:
            self.found_offsets.append(offset)
            found_point = np.array([0, offset])
            self.found_points.append([points, interpolation_for_plot, found_point, np.sign(offset)])

        return offset

    @staticmethod
    def _linear_interpolate(points):
        return interpolate.interp1d(points[:, 0], points[:, 1], kind="linear")

    def project_weno(self, points, are_local=False):
        n_points = 32
        points = np.array(points, dtype=float)
        if not are_local:
            points = self._to_local_coordinate_system(points, n_closest=n_points)
        if not self.low_memory:
            x_s = np.linspace(min(points[:, 0]), max(points[:, 0]), 1000)
            interpolation_y = weno4(xs=x_s, xp=points[:, 0], fp=points[:, 1])
            offset = interpolation_y[np.abs(x_s).argmin()]
            interpolation_for_plot = np.stack((x_s,
                                               interpolation_y))
            found_point = np.array([0, offset])
            self.found_points.append([points, interpolation_for_plot, found_point, np.sign(offset)])
        else:
            x_s = 0
            offset = weno4(xs=x_s, xp=points[:, 0], fp=points[:, 1])

        simpler_proj = self._linear_interpolate(points)(0)
        tolerance = 2
        if np.abs(offset - simpler_proj) > tolerance:
            offset = simpler_proj
        return offset


class PerpendicularAdjuster(PerpendicularProj):
    def __init__(self, mid_pt, second_pt):
        super(PerpendicularAdjuster, self).__init__(mid_pt, second_pt)
        self.found_adjustments = []
        self.image_processor = ImageProcessor()

    def get_profile_around_detection(self, image, detection, radius):
        self.image_processor.load_image(image)
        pt1 = self.get_point_at_dist(radius + detection)
        pt2 = self.get_point_at_dist(-radius + detection)
        profile_line = self.image_processor.get_profile(pt2, pt1)
        profile_line = profile_line[(len(profile_line) - 2 * radius):]
        x_pixel = np.arange(-radius, radius)
        x_fine = np.linspace(-radius, radius, 10000)

        return x_pixel, x_fine, profile_line

    def adjust_devernay(self, frame_image, radius=20, rough=0, preprocess_locally=True):
        x_pixel, x_fine, profile_line = self.get_profile_around_detection(frame_image,
                                                                          rough,
                                                                          radius,
                                                                          preprocess_locally)

        gradient_norm_of_profile = np.gradient(profile_line)
        plt.plot(profile_line)
        plt.plot(gradient_norm_of_profile)
        plt.show()
        g_B = gradient_norm_of_profile[profile_line.size//2]  # see vonGioi2017, section 3
        g_A = gradient_norm_of_profile[(profile_line.size // 2) - 1]
        g_C = gradient_norm_of_profile[(profile_line.size // 2) + 1]

        adjustment = .5 * (g_A - g_C) / (g_A + g_C - 2*g_B)

        if not self.low_memory:
            profile_line_plot = np.stack((x_pixel,
                                          profile_line))
            fit_plot = np.stack((x_pixel,
                                 gradient_norm_of_profile))
            self.found_adjustments.append([profile_line_plot, adjustment, fit_plot, 0])

        return adjustment

    def adjust_tanh(self, frame_image, radius=20, rough=0, preprocess_locally=True):
        def tanh(x, a, eta, phi, b):
            return a * np.tanh(eta * (x + phi)) + b

        if preprocess_locally:
            frame_image, _ = self.get_local_image(frame_image, radius=radius, offset_to_mid=rough)
            self.image_processor.load_image(frame_image)
            self.image_processor.global_hist_equal()
            pt1 = radius * (1 + self.get_perp_direction())
            pt2 = radius * (1 - self.get_perp_direction())
        else:
            self.image_processor.load_image(frame_image)
            pt1 = self.get_point_at_dist(radius + rough)
            pt2 = self.get_point_at_dist(-radius + rough)

        profile_line = self.image_processor.get_profile(pt2, pt1)
        profile_line = profile_line[(len(profile_line) - 2 * radius):]
        x_pixel = np.arange(-radius, radius)
        x_fine = np.linspace(-radius, radius, 10000)
        try:
            fit_params, var_matrix = optimize.curve_fit(tanh, x_pixel, profile_line)

            adjustment = x_fine[np.argmax(np.absolute(np.gradient(tanh(x_fine, *fit_params))))]

            if not self.low_memory:
                accuracy_std = np.sqrt(np.diag(var_matrix))
                profile_line_plot = np.stack((x_pixel,
                                              profile_line))
                fit_plot = np.stack((x_fine,
                                     tanh(x_fine, *fit_params)))
                self.found_adjustments.append([profile_line_plot, adjustment, fit_plot, accuracy_std])

        except Exception:
            adjustment = 0
        # if adjustment > 10:
        #     print(self.mid, rough, pt1, pt2, np.sqrt(np.diag(var_matrix)))
        #     fig, ax = plt.subplots(2)
        #     ax[0].imshow(self.image_processor.result(), cmap="gray")
        #     ax[0].plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
        #     ax[0].scatter(self.mid[0], self.mid[1])
        #     rough_point = self.get_point_at_dist(rough)
        #     ax[0].scatter(rough_point[0], rough_point[1])
        #     ax[1].plot(x_pixel, profile_line)
        #     ax[1].plot(x_fine, tanh(x_fine, *fit_params))
        #     print(adjustment)
        #     ax[1].axvline(adjustment, linestyle='--', color='r')
        #     ax[1].axvline(0, linestyle='-', color='blue')
        #     plt.show()
        #     plt.close()
        return adjustment


class SmartPerpendicular(PerpendicularAdjuster):
    def project(self, points, frame_image):
        try:
            detection = self.project_weno(points)
        except Exception as e:
            detection = self.project_interpolation(points)
        adjusted_detection = self.adjust_devernay(frame_image, rough=detection)

        return adjusted_detection


def test_perpendicular_proj():
    perp = PerpendicularAdjuster(mid_pt=np.array([1, 1]), second_pt=np.array([0, 0]))
    point_to_project = np.array([-1, -3])
    point_to_project_2 = np.array([0, -1])
    point_to_project_3 = np.array([1, 3])
    points = np.stack([point_to_project,
                       point_to_project_2,
                       point_to_project_3])
    projected_points = perp._to_local_coordinate_system(points)
    print(projected_points)
