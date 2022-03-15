from abc import ABC

import cv2
import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt
from tifffile import tifffile
import matplotlib.figure as fig
import io as sysio


class ImageRecorder(ABC):
    def save(self, path, image):
        if path.endswith('.png'):
            self._save_in_png(path, image)
        elif path.endswith('.tif'):
            self._save_in_tiff(path, image)
        else:
            raise TypeError

    def _interpreter(self, image):
        if isinstance(image, np.ndarray):
            return np.expand_dims(image, axis=-1)
        if isinstance(image, fig.Figure):
            in_memory = sysio.BytesIO()
            image.savefig(in_memory, format="png")
            return np.asarray(Image.open(in_memory))

    def _save_in_tiff(self, tiffpath, image):
        with tifffile.TiffWriter(tiffpath, append=True) as stack:
            if isinstance(image, list):
                for frame in image:
                    stack.save(self._interpreter(frame), contiguous=False)
            else:
                stack.save(self._interpreter(image), contiguous=False)
        del image

    def _save_in_png(self, pngpath, image):
        if isinstance(image, fig.Figure):
            plt.tight_layout()
            plt.savefig(pngpath, bbox_inches='tight')
            plt.close()
        elif isinstance(image, np.ndarray):
            cv2.imwrite(pngpath, image)
