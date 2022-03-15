from edge_processing import EdgeAnalyzer
from raw_reader import RawReader
from systematic_stuff.fluctuations.boundary_analysis import load_pickle
from systematic_stuff.convenience_functions import _preprocess_image
import numpy as np
import os
import matplotlib.pyplot as plt

EDGE = 'edge 1'
REGION = '3microns, 0.2s, r2'
# SRC_FOLDER = os.path.join(r"C:\Users\User\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations", REGION)

SRC_FOLDER = os.path.join(r"C:\Users\c1545871\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations", REGION)
# TARGET_FOLDER = os.path.join(r"E:\results", REGION)
TARGET_FOLDER = os.path.join(r"E:\results", REGION)
RESULTS_PATH = os.path.join(TARGET_FOLDER, EDGE, 'manual_mask')


def load_images():
    read = RawReader()
    images = read.read_folder(SRC_FOLDER)
    for image in images:
        image.data = np.stack((_preprocess_image(image.data),) * 3, axis=-1)

    return images


def load_edges():
    return load_pickle(os.path.join(RESULTS_PATH, 'coordinates.pickle'))


def load_offsets():
    return load_pickle(os.path.join(RESULTS_PATH, 'offsets.pickle'))


def get_perps(edges):
    anal = EdgeAnalyzer(edges)
    anal.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=8)
    return anal.perpendiculars


def draw_edges(images, edges):
    color = np.array([0, 255, 0])
    alpha = 0.5
    for frame, image in enumerate(images):
        image = image.data
        edge_coords = edges[frame]
        images[frame].data[edge_coords[:, 0], edge_coords[:, 1]] = (1 - alpha) * image[
            edge_coords[:, 0], edge_coords[:, 1]] + alpha * color


def test_edges(edges):
    anal = EdgeAnalyzer(edges)
    edge = anal.edges[0]
    anal.get_perpendiculars_adaptive(edge_frame=0, every_point=True, pixel_average=8)
    perp_coord = np.array([(i.mid[0], i.mid[1]) for i in anal.perpendiculars])
    plt.figure()
    plt.plot(edge[:, 0], edge[:, 1])
    plt.scatter(edge[:, 0], edge[:, 1])

    plt.scatter(perp_coord[:, 0], perp_coord[:, 1])
    plt.show()
    plt.close()


def draw_all_else(images, perps, offsets):
    def point_true_coords(mid, second, dist_to_mid):
        def dy(distance, m):
            return m * dx(distance, m)

        def dx(distance, m):
            return np.sqrt(distance / (m ** 2 + 1))

        a = (second[1] - mid[1]) / (second[0] - mid[0])
        b = mid[1] - a * mid[0]
        point = (mid[0] + dx(dist_to_mid, a), mid[1] + dy(dist_to_mid, a))
        other_possible_point_b = (mid[0] - dx(dist_to_mid, a), mid[1] - dy(dist_to_mid, a))
        return point

    for frame, (image, offsets) in enumerate(zip(images, offsets)):
        plt.figure()
        plt.imshow(image.data)
        plt.axis("equal")
        for perp, offset_pt in zip(perps, offsets):
            detected_point = point_true_coords(perp.mid, perp.second, offset_pt)
            plt.scatter(perp.mid[1], perp.mid[0], color='black', s=1)  # the midpoints of fit lines
            plt.scatter(detected_point[1], detected_point[0], color='red')
            plt.axline(perp.mid[::-1], perp.second[::-1], linestyle='dashed', alpha=0.6)  # the perpendiculars
        savepath = os.path.join(TARGET_FOLDER, "for_juan")
        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, "frame_{}.png".format(frame))
        plt.xlim(350, 500)
        plt.ylim(500, 650)
        plt.axis('off')
        plt.savefig(savepath)
        plt.close()


if __name__ == '__main__':
    images = load_images()
    edges = load_edges()
    offsets = load_offsets()
    perps = get_perps(edges)

    draw_edges(images, edges)
    draw_all_else(images, perps, offsets)
