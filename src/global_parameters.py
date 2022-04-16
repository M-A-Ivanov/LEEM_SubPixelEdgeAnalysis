import seaborn as sns
from operator import sub

sns.set_theme()
# sns.set_style("whitegrid")
sns.set_context('paper', font_scale=0.9)

pixel = 6000 / 1024  # pixel size in nm

CANNY_SIGMA = 1.2  # pixels

MASK_SIZE = 8  # pixels in radius

HIST_EQUAL = "NORMAL"


class Constants:
    """
        k_B: meV/K, Boltzmann const
        T: K, temperature
        w: nm, terrace width
        M: ev/A^3, Young's modulus, GaAs
        ni: Poisson ratio, GaAs
        M_Si: ev/A^3, Young's modulus, Si
        ni_Si: Poisson ratio, Si
    """
    k_B = 8.617e-2  # meV/K, Bolzmann const
    T = 550 + 273.15  # K, temperature
    w = 220  # nm, terrace width
    M = 0.53  # ev/A^3, Young's modulus, GaAs
    M_Si = 1  # --------||-------------, Si
    ni = 0.31  # Poisson ratio, GaAs
    ni_Si = 0.27  # ---||-----, Si


class AxisNames:

    @staticmethod
    def fft():
        return {"x": r"$q^{2}}$, $nm^{-2}$",
                "y": r"$\dfrac{1}{\langle |y_{q}|^{2}\rangle}$, $nm^{-2}$"}

    @staticmethod
    def distr():
        return {"x": r"Offsets $\Delta x$, nm",
                "y": r"Probability"}


class FigureControl:
    column_width = 4
    page_width = 8
    page_height = 11
    from operator import sub

    @staticmethod
    def get_aspect(ax):
        # Total figure size
        figW, figH = ax.get_figure().get_size_inches()
        # Axis size on figure
        _, _, w, h = ax.get_position().bounds
        # Ratio of display units
        disp_ratio = (figH * h) / (figW * w)
        # Ratio of data units
        # Negative over negative because of the order of subtraction
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

        return disp_ratio / data_ratio

    def forceAspect(self, ax, aspect):
        im = ax.get_images()
        extent = im[0].get_extent()
        ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

    def for_paper(self, figure, ax, page_width_frac=1):
        w = self.page_width*page_width_frac
        h = self.get_aspect(ax) * w
        figure.set_size_inches(w, h)


class SuccessTracker:
    successful = 0
    unsuccessful = 0

    def reset(self):
        self.successful = 0
        self.unsuccessful = 0

    def success(self):
        self.successful += 1

    def failure(self):
        self.unsuccessful += 1

    def total(self):
        return self.successful + self.unsuccessful

    def success_rate(self):
        rate = self.successful / self.total()
        print("Success rate: {} %".format(rate * 100))
        return rate

    def failure_rate(self):
        rate = self.unsuccessful / self.total()
        print("Failure rate: {} %".format(rate * 100))
        return rate

