import io
from typing import List

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib

from global_paths import REGION, EDGE, SRC_FOLDER, TARGET_FOLDER
from image_processing import ImageProcessor
from raw_reader import RawReader
from systematic_stuff.fluctuations.boundary_detection import FluctuationsDetector
import os


def main():
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        # Title of Window
        [sg.Text("Boundary Detection", size=(60, 1), justification="center")],
        # Where the main plot is shown
        [sg.Image(key="-CANVAS-")],

        # Where to put number of frame to use as example
        [[sg.Text('Choose nubmer of frames to use as example'), sg.InputText("10", key='-N FRAMES-'),
          sg.Button("Load example")]],
        [sg.Slider((0, 10), key="-SEL FRAME-", enable_events=True)],

        # Mask controls
        [sg.Radio('Load previous mask', 0, key="-LOAD MASK-", default=True),
         [sg.Radio('Create new mask', 0, key="-NEW MASK-", default=False),
          [sg.Text('Mask Size (integer):'), sg.InputText("8", key='-MASK SIZE-')],
          sg.Checkbox('Save new mask', default=True)]],

        # Choose Hist Equal mode
        [sg.Radio('Global Histogram Equalization', 1, key= "-GLOBAL HIST-", default=False),
         sg.Radio('CLAHE', 1, key="-CLAHE HIST-", default=True)],

        # Choose Denoising method
        [sg.Radio('NLM', 2, key="-NLM DNS-", default=True),
         sg.Radio('Bilateral Filtering', 2, key="-BILAT DNS-", default=False),
         sg.Radio('Wavelet', 2, key="-WVL DNS-", default=False),
         sg.Radio('TV', 2, key="-TV DNS-", default=False)],

        # Canny edge control
        [sg.Slider(
            (0, 12),
            1,
            0.2,
            orientation="h",
            size=(20, 15),
            key="-CANNY SIGMA SLIDER-",
        )
        ],
        [sg.Button("Apply"),
         sg.Button("Do for all frames and save")]
    ]

    window = sg.Window("OpenCV Integration", layout, location=(800, 400))

    def draw_figure(element, figure):
        """
        Draws the previously created "figure" in the supplied Image Element
        :param element: an Image Element
        :param figure: a Matplotlib figure
        :return: The figure canvas
        """

        plt.close('all')  # erases previously drawn plots
        canv = FigureCanvasAgg(processors[selected_frame].figure_all())
        buf = io.BytesIO()
        canv.print_figure(buf, format='png')
        if buf is not None:
            buf.seek(0)
            element.update(data=buf.read())
            return canv
        else:
            return None

    def generate_figures(image_processors: List[ImageProcessor]):
        plt.close("all")
        for processor in image_processors:
            if len(processor.images)>1:
                processor.revert(4)
            if values["-CLAHE HIST-"]:
                processor.clahe_hist_equal()
            if values["-GLOBAL HIST-"]:
                processor.clahe_hist_equal()
            if values["-NLM DNS-"]:
                processor.denoise_nlm(fast=True)
            if values["-BILAT DNS-"]:
                processor.denoise_bilateral()
            if values["-WVL DNS-"]:
                processor.denoise_wavelet()
            if values["-TV DNS-"]:
                processor.denoise_tv()
            processor.canny_devernay_edges(mask=mask, sigma=values["-CANNY SIGMA SLIDER-"])
            processor.clean_up(5)

    reader = RawReader()
    processors = [ImageProcessor() for _ in range(10)]
    images = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=(0, 10))

    for processor in processors:
        processor.load_image(images[0].data)
        processor.align(preprocess=False)
    # Main loop
    while True:
        event, values = window.read(timeout=20)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        # Add the plot to the window

        num_frames = int(values["-N FRAMES-"])
        selected_frame = int(values["-SEL FRAME-"])

        if event == "Load example":
            num_frames = int(values["-N FRAMES-"])
            selected_frame = int(values["-SEL FRAME-"])
            images = reader.read_single(os.path.join(SRC_FOLDER, REGION), frames=(0, num_frames))
            processors = [ImageProcessor() for _ in range(num_frames)]
            window["-SEL FRAME-"].Update(range=(0, num_frames))
            for processor in processors:
                processor.load_image(images[0].data)
                processor.align(preprocess=False)

            draw_figure(window["-CANVAS-"], processors[selected_frame])

        if event == "-SEL FRAME-":
            selected_frame = int(values["-SEL FRAME-"])
            draw_figure(window["-CANVAS-"], processors[selected_frame])

        if event == "Apply":
            draw_figure(window["-CANVAS-"], processors[selected_frame])
            window['-CANVAS-'].update(visible=True)
            generate_figures(processors)

    window.close()


if __name__ == '__main__':
    main()
