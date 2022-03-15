from image_processing import ImageProcessor
from raw_reader import RawReader
import os


FOLDER = r"C:\Users\User\OneDrive - Cardiff University\Data\LEEM\27042021\fluctuations\3microns, 0.2s, r1"


if __name__ == '__main__':
    path = os.path.join(FOLDER, 'aligned')
    os.makedirs(path, exist_ok=True)
    reader = RawReader()
    processor = ImageProcessor()
    list_of_images = reader.read_folder(FOLDER)
    for image in list_of_images[:500]:
        processor.load_image(image.data)
        processor.align()
        processor.save_last(savefig=os.path.join(path, 'aligned.tif'))
