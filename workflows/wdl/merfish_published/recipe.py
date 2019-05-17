import numpy as np
from copy import deepcopy

import starfish
from starfish import FieldOfView
from starfish.types import Features, Axes
from starfish.image import Filter
from starfish.types import Clip
from starfish.spots import DetectPixels


def process_fov(field_of_view, experiement_str):
    fov_str: str = f"fov_{int(field_of_view):03d}"
    # load experiment
    experiment = starfish.Experiment.from_json(experiement_str)

    print(f"loading fov: {fov_str}")
    fov = experiment[fov_str]
    imgs = fov.get_images(FieldOfView.PRIMARY_IMAGES)

    print("gaussian high pass")
    ghp = Filter.GaussianHighPass(sigma=3)
    high_passed = ghp.run(imgs, verbose=True, in_place=False)

    print("deconvoling")
    dpsf = Filter.DeconvolvePSF(num_iter=15, sigma=2, clip_method=Clip.SCALE_BY_CHUNK)
    deconvolved = dpsf.run(high_passed, verbose=True, in_place=False)

    print("guassian low pass")
    glp = Filter.GaussianLowPass(sigma=1)
    low_passed = glp.run(deconvolved, in_place=False, verbose=True)

    scale_factors = {
        (t[Axes.ROUND], t[Axes.CH]): t['scale_factor']
        for t in experiment.extras['scale_factors']
    }
    filtered_imgs = deepcopy(low_passed)

    for selector in imgs._iter_axes():
        data = filtered_imgs.get_slice(selector)[0]
        scaled = data / scale_factors[selector[Axes.ROUND.value], selector[Axes.CH.value]]
        filtered_imgs.set_slice(selector, scaled, [Axes.ZPLANE])

    print("decoding")
    psd = DetectPixels.PixelSpotDecoder(
        codebook=experiment.codebook,
        metric='euclidean',  # distance metric to use for computing distance between a pixel vector and a codeword
        norm_order=2,  # the L_n norm is taken of each pixel vector and codeword before computing the distance. this is n
        distance_threshold=0.5176,  # minimum distance between a pixel vector and a codeword for it to be called as a gene
        magnitude_threshold=1.77e-5,  # discard any pixel vectors below this magnitude
        min_area=2,  # do not call a 'spot' if it's area is below this threshold (measured in pixels)
        max_area=np.inf,  # do not call a 'spot' if it's area is above this threshold (measured in pixels)
    )

    initial_spot_intensities, prop_results = psd.run(filtered_imgs)

    spot_intensities = initial_spot_intensities.loc[initial_spot_intensities[Features.PASSES_THRESHOLDS]]
    df = spot_intensities.to_decoded_spots()
    return df