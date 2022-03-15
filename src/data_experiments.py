import os

from global_paths import SRC_FOLDER, REGION
from edge_segmentation import EdgeContainer
from image_processing import ImageProcessor, average_images, average_image_list
from raw_reader import RawReader
from tqdm import tqdm


def experimental_detection():
    average = 5
    reader = RawReader()
    processor = ImageProcessor()
    image_align = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=0).data
    processor.load_image(image_align)
    processor.align(preprocess=True)
    if average == 1:
        image = reader.read_single(os.path.join(SRC_FOLDER, REGION)).data
    else:
        images = [img.data for img in reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=(50, 50+average))]
        image = average_images(images)
    processor.load_image(image)
    processor.align(preprocess=True)
    mask = processor.get_mask(15)
    mask = processor.cut_to_mask(mask)
    processor.global_hist_equal()
    processor.denoise_nlm(fast=True)
    try:
        processor.canny_edges(sigma=2.25, mask=mask)
        processor.clean_up(25)
    except Exception as e:
        print("Canny fails with: {}".format(e))
    processor.figure_all()


def load_experimental_images(folder_src, num_images=None):
    reader = RawReader()
    if num_images is None:
        list_of_images = reader.read_folder(folder_src)
    else:
        list_of_images = reader.read_single(folder_src, frames=(0, num_images))
    list_of_images = [image.data for image in list_of_images]
    processor = ImageProcessor()
    processor.load_image(list_of_images[0])
    mask = processor.get_mask(15)
    return list_of_images, mask


def experimental_manual_masking(list_of_images, mask=None, align=True,
                                average=1):
    processor = ImageProcessor()
    list_of_images = average_image_list(list_of_images, average)
    processor.load_image(list_of_images[0])
    if align:
        processor.align(preprocess=True)
    if mask is None:
        original_mask = processor.get_mask(15)
    else:
        original_mask = mask
    grouper_canny = EdgeContainer()
    print("Found images: {}".format(len(list_of_images)))
    for i, image in enumerate(tqdm(list_of_images)):
        try:
            processor.load_image(image)
            if align:
                processor.align(preprocess=True)
            mask = processor.cut_to_mask(original_mask)
            processor.global_hist_equal()
            processor.denoise_nlm(fast=True)
            processor.canny_edges(mask=mask, sigma=2.5)
            processor.clean_up(25)
            grouper_canny.append_edges(processor.edge_result())
        except Exception as e:
            print("Frame {} failed with: {}".format(i, e))

    return grouper_canny.get_coordinates()


if __name__ == "__main__":
    experimental_detection()