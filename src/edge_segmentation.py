import pickle

from image_recorder import ImageRecorder
import numpy as np
from skimage import measure
from skimage import draw
from skimage import morphology
import matplotlib.pyplot as plt
import os


class CannyEdge:
    """This is placed directly onto image, having pixel-to-pixel correspondence"""
    def __init__(self, image, edge):
        self.edge = edge
        self.image = image


class EdgeContainer(ImageRecorder):
    def __init__(self):
        self.edges = []

    def append_edges(self, edges):
        """ Alias to append_coordinates() """
        self.append_coordinates(edges)

    def append_coordinates(self, coordinates):
        self.edges.append(coordinates)

    def save_coordinates(self, path, suffix=" "):
        coords = [np.squeeze(np.array(edge)) for edge in self.edges]
        picklepath = os.path.join(path, "coordinates_{}.pickle".format(suffix))
        with open(picklepath, "wb") as f:
            pickle.dump(coords, f)

    def load_coordinates(self, path):
        picklepath = os.path.join(path, "coordinates.pickle")
        with open(picklepath, "rb") as f:
            ret = pickle.load(f)

        return ret

    def get_coordinates(self):
        return [np.squeeze(np.array(edge)) for edge in self.edges]

    def save_raw_edges(self, path):
        picklepath = os.path.join(path, "raw_edges.pickle")
        with open(picklepath, "wb") as f:
            pickle.dump(self.edges, f)

    def load_raw_edges(self, path):
        picklepath = os.path.join(path, "raw_edges.pickle")
        with open(picklepath, "rb") as f:
            self.edges = pickle.load(f)

    def save_images(self, path, label=0, time=None, as_tiff=False):
        if as_tiff:
            end = '.tif'
        else:
            end = '.png'

        subfolder = os.path.join(path, str(label))
        os.makedirs(subfolder, exist_ok=True)

        for i, edge in enumerate(self.edges):
            fig = plt.figure()
            plt.plot(edge[:, 0], edge[:, 1])
            plt.scatter(edge[:, 0], edge[:, 1])
            if time is not None:
                plt.title("Edge {} at {:.3f}s".format(int(label), time))
            else:
                plt.title("Edge {}".format(int(label)))
            if as_tiff:
                self.save(os.path.join(subfolder, 'edges' + end), fig)
            else:
                self.save(os.path.join(subfolder, str(i) + end), fig)
            plt.close()


class ImageMasker:
    def __init__(self):
        self.image = None
        self.mask = None

    def load_image(self, img):
        self.image = img

    def get_manual_points(self):
        coords = []

        def onclick(event):
            coords.append((round(event.xdata, 3), round(event.ydata, 3)))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(self.image, cmap='gray')
        for i in range(0, 1):
            cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()
        return coords

    def get_manual_mask(self, width, show=False):
        coords = self.get_manual_points()
        mask = np.zeros_like(self.image, dtype=bool)
        for A, B in zip(coords[:-1], coords[1:]):
            rr, cc = draw.line(int(A[0]), int(A[1]), int(B[0]), int(B[1]))
            mask[cc, rr] = 1

        disk = morphology.disk(radius=width)
        mask = morphology.dilation(mask, disk)
        if show:
            plt.figure()
            plt.imshow(mask)
            plt.show()
            plt.close()
        return mask




