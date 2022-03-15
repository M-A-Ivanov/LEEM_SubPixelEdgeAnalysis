import numpy as np
from numpy import ndarray
from scipy.ndimage import generate_binary_structure
from skimage import img_as_float, dtype_limits
import scipy.ndimage as ndi
from scipy.ndimage import generate_binary_structure, binary_erosion, label
from skimage.filters import gaussian
from skimage._shared.utils import check_nD


def _preprocess(image: ndarray, mask: ndarray, sigma, mode, cval):
    """Generate a smoothed image and an eroded mask.

    The image is smoothed using a gaussian filter ignoring masked
    pixels and the mask is eroded.

    Parameters
    ----------
    image : array
        Image to be smoothed.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.

    Returns
    -------
    smoothed_image : ndarray
        The smoothed array
    eroded_mask : ndarray
        The eroded mask.

    Notes
    -----
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """

    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval,
                           preserve_range=False)
    if mask is None:
        # Smooth the masked image
        smoothed_image = gaussian(image, **gaussian_kwargs)
        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
        return smoothed_image, eroded_mask

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Compute the fractional contribution of masked pixels by applying
    # the function to the mask (which gets you the fraction of the
    # pixel data that's due to significant points)
    bleed_over = (
            gaussian(mask.astype(float), **gaussian_kwargs) + np.finfo(float).eps
    )

    # Smooth the masked image
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)

    # Lower the result by the bleed-over fraction, so you can
    # recalibrate by dividing by the function on the mask to recover
    # the effect of smoothing from just the significant pixels.
    smoothed_image /= bleed_over

    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    return smoothed_image, eroded_mask


def canny_devernay(image, sigma=1., low_threshold=None, high_threshold=None, mask=None,
                   use_quantiles=False):
    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not (0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold = low_threshold / dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not (0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold = high_threshold / dtype_max

    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode='constant', cval=1)
    jsobel = ndi.sobel(smoothed, axis=1)
    isobel = ndi.sobel(smoothed, axis=0)
    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)
    magnitude = np.hypot(isobel, jsobel)

    #
    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    #

    eroded_mask = eroded_mask & (magnitude > 0)
    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(image.shape, bool)
    correction_matrix_i = np.zeros(image.shape)  # mine: correction in i and j, e.g. subpixel adjustment
    correction_matrix_j = np.zeros(image.shape)
    # ----- 0 to 45 degrees ------
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    g_plus = c2 * w + c1 * (1 - w)
    c_plus = g_plus <= m
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    g_minus = c2 * w + c1 * (1 - w)
    c_minus = g_minus <= m
    local_maxima[pts] = c_plus & c_minus
    devernay_correction = .5 * (g_plus - g_minus) / (g_plus + g_minus - 2 * m)
    correction_matrix_i[pts] = devernay_correction * (isobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    correction_matrix_j[pts] = devernay_correction * (jsobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts_plus = (isobel >= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel <= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:, 1:][pts[:, :-1]]
    c2 = magnitude[1:, 1:][pts[:-1, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    g_plus = c2 * w + c1 * (1 - w)
    c_plus = g_plus <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[:-1, :-1][pts[1:, 1:]]
    g_minus = c2 * w + c1 * (1 - w)
    c_minus = g_minus <= m
    devernay_correction = .5 * (g_plus - g_minus) / (g_plus + g_minus - 2 * m)
    local_maxima[pts] = c_plus & c_minus
    correction_matrix_i[pts] = devernay_correction * (isobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    correction_matrix_j[pts] = devernay_correction * (jsobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))

    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel <= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel <= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1a = magnitude[:, 1:][pts[:, :-1]]
    c2a = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_isobel[pts] / abs_jsobel[pts]
    g_plus = c2a * w + c1a * (1.0 - w)
    c_plus = g_plus <= m
    c1 = magnitude[:, :-1][pts[:, 1:]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    g_minus = c2 * w + c1 * (1.0 - w)
    c_minus = g_minus <= m
    devernay_correction = .5 * (g_plus - g_minus) / (g_plus + g_minus - 2 * m)
    local_maxima[pts] = c_plus & c_minus
    correction_matrix_i[pts] = devernay_correction * (isobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    correction_matrix_j[pts] = devernay_correction * (jsobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts_plus = (isobel <= 0) & (jsobel >= 0) & (abs_isobel >= abs_jsobel)
    pts_minus = (isobel >= 0) & (jsobel <= 0) & (abs_isobel >= abs_jsobel)
    pts = pts_plus | pts_minus
    pts = eroded_mask & pts
    c1 = magnitude[:-1, :][pts[1:, :]]
    c2 = magnitude[:-1, 1:][pts[1:, :-1]]
    m = magnitude[pts]
    w = abs_jsobel[pts] / abs_isobel[pts]
    g_plus = c2 * w + c1 * (1 - w)
    c_plus = g_plus <= m
    c1 = magnitude[1:, :][pts[:-1, :]]
    c2 = magnitude[1:, :-1][pts[:-1, 1:]]
    g_minus = c2 * w + c1 * (1 - w)
    c_minus = g_minus <= m
    devernay_correction = .5 * (g_plus - g_minus) / (g_plus + g_minus - 2 * m)
    local_maxima[pts] = c_plus & c_minus
    correction_matrix_i[pts] = devernay_correction * (isobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))
    correction_matrix_j[pts] = devernay_correction * (jsobel[pts] /
                                                      (np.sqrt(isobel[pts] * isobel[pts] + jsobel[pts] * jsobel[pts])))

    #
    # ---- If use_quantiles is set then calculate the thresholds to use
    #
    if use_quantiles:
        high_threshold = np.percentile(magnitude, 100.0 * high_threshold)
        low_threshold = np.percentile(magnitude, 100.0 * low_threshold)

    #
    # ---- Create two masks at the two thresholds.
    #
    high_mask = local_maxima & (magnitude >= high_threshold)
    low_mask = local_maxima & (magnitude >= low_threshold)

    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    strel = np.ones((3, 3), bool)
    labels, count = label(low_mask, strel)
    if count == 0:
        return low_mask

    sums = (np.array(ndi.sum(high_mask, labels,
                             np.arange(count, dtype=np.int32) + 1),
                     copy=False, ndmin=1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums > 0
    output_mask = good_label[labels]
    indeces = np.nonzero(output_mask)
    coordinates = np.stack((correction_matrix_i[indeces] + indeces[0], correction_matrix_j[indeces] + indeces[1]),
                           axis=1)
    assert (-0.5 < correction_matrix_i[indeces]).all() & (correction_matrix_i[indeces] < 0.5).all()
    assert (-0.5 < correction_matrix_j[indeces]).all() & (correction_matrix_j[indeces] < 0.5).all()

    return output_mask, coordinates
