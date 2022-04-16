import os
import numpy as np
from DATImage import DATImage
import random
from natsort import natsorted
import tifffile as tiff


class TIFFImage:
    def __init__(self, file_path):
        self.data = tiff.imread(file_path)


class RawReader:
    def read_folder_simple(self, folder_path):
        file_list = os.listdir(folder_path)
        list_of_images = list()
        for i, file in enumerate(natsorted(file_list)):
            if file.endswith('.dat'):
                list_of_images.append(self.read_simple(os.path.join(folder_path, file)))

        return np.array(list_of_images)

    @staticmethod
    def read_simple(file_path):
        offset = abs(2 ** 21 - os.path.getsize(file_path))
        with open(file_path, 'r') as f:
            f.seek(offset)
            return np.fromfile(f, dtype=np.uint16).reshape((1024, 1024))

    def read_single(self, folder_path, frames=None):
        file_list = os.listdir(folder_path)
        file_list = [file for file in file_list if file.endswith('.dat') or file.endswith('.tif')]
        file_list = natsorted(file_list)
        if frames is None:
            choice = random.choice(file_list)
            return self.read(os.path.join(folder_path, choice))
        else:
            if isinstance(frames, int):
                choice = file_list[frames]
                return self.read(os.path.join(folder_path, choice))
            elif isinstance(frames, tuple):
                return [self.read(os.path.join(folder_path, file_list[frame])) for frame in range(frames[0], frames[1])]

    @staticmethod
    def read(file_path):
        if file_path.endswith('.dat'):
            return DATImage(file_path)
        elif file_path.endswith('.tif'):
            return TIFFImage(file_path)

    def read_folder(self, folder_path, first_n=None, every_nth=None):
        file_list = os.listdir(folder_path)
        file_list = natsorted(file_list)
        if first_n is not None:
            if first_n > 0:
                file_list = file_list[:first_n]
            else:
                file_list = file_list[first_n:]
        if every_nth is not None:
            file_list = file_list[::every_nth]
        list_of_images = list()
        for i, file in enumerate(file_list):
            if file.endswith('.dat') or file.endswith('.tif'):
                list_of_images.append(self.read(os.path.join(folder_path, file)))
        return list_of_images

    def read_folder_conditional(self, folder_path, FOV=None, time=None):
        """In development, don't use!"""
        file_list = os.listdir(folder_path)
        list_of_images = list()
        for i, file in enumerate(file_list):
            if file.endswith('.dat'):
                list_of_images.append(self.read(os.path.join(folder_path, file)))
        return list_of_images

    def _check_conditions(self, img: DATImage, FOV, time):
        truth = True
        if FOV is not None:
            truth = truth and (img.metadata['FOV'] == FOV)
        if time is not None:
            if isinstance(time, list):
                pass


if __name__ == '__main__':
    folder = "C:\\Users\\User\\OneDrive - Cardiff University\\Data\\LEEM\\03032021\\high FPS imaging 0.2s\\data"

    reader = RawReader()
    images = reader.read_single(folder)
    print(images.metadata)

