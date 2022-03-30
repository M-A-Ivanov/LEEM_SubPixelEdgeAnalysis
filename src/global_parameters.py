import seaborn as sns

sns.set_theme()
sns.set_style("whitegrid")
sns.set_context('paper', font_scale=1.2)

pixel = 6000 / 1024  # pixel size in nm

CANNY_SIGMA = 0.75  # pixels

MASK_SIZE = 8  # pixels in radius

HIST_EQUAL = "NORMAL"


class AxisNames:

    @staticmethod
    def fft():
        return {"x": r"$q^{2}}$, $nm^{-2}$",
                "y": r"$\frac{1}{\langle |y_{q}|^{2}\rangle}$, $nm^{-2}$"}

    @staticmethod
    def distr():
        return {"x": r"Offsets $\Delta x$, nm",
                "y": r"Probability"}


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

