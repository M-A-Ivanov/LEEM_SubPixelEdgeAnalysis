from skimage.feature import blob_log

import matplotlib.pyplot as plt
import os
import tifffile
from image_processing import ImageProcessor
import random


DATA_PATH = r"F:\cardiff cloud\OneDrive - Cardiff University\Data\LEEM\11092019\experiment\recording\growth\2098521"
FILE_PATH = os.path.join(DATA_PATH, "cropped and clean small cut and aligned.tif")


class BlobDetector(ImageProcessor):
    def __init__(self):
        super(BlobDetector, self).__init__()
        self.blob_stats = []

    def detect_blobs(self):
        images = self.images[-1]
        for image in images:
            self.blob_stats.append(blob_log(image, max_sigma=10, num_sigma=10, threshold=.1))

    def show_example_blobs(self):
        index = random.randint(0, len(self.blob_stats)-1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.images[-1][index])
        for blob in self.blob_stats[index]:
            y, x, z, r = blob
            ax.add_patch(plt.Circle((x, y), r, color='r', linewidth=2, fill=False))
        plt.tight_layout()
        plt.show()
        plt.close()


def init_testing():
    with tifffile.TiffFile(FILE_PATH) as tifffffff:
        images = tifffffff.asarray()

    process = BlobDetector()
    process.load_image(images)
    process.global_hist_equal()
    process.invert()
    process.detect_blobs()
    process.show_example_blobs()


if __name__ == '__main__':
    init_testing()
