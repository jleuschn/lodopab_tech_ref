# -*- coding: utf-8 -*-
"""
This script is a technical reference how the `LoDoPaB-CT dataset
<https://doi.org/10.5281/zenodo.3384092>`_ was created.
The dataset is described in an arXiv paper in more detail.

Prerequisites:

    * imported libraries + ``astra-toolbox==1.8.3`` (requires Python 3.6)
    * LIDC-IDRI dataset stored in `DATA_PATH`
      (available from `TCIA
      <https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI>`_)

Note: at the moment, the file ``lidc_idri_file_list.json`` is a dummy file
created in the same way as the one for LoDoPaB-CT, but with different
randomization. This is done in order to keep private which data is used for the
"challenge" part. Also, the random seeds used in this script are modified.
"""
import os
import json
from itertools import islice
from math import ceil
import numpy as np
import odl
from tqdm import tqdm
from skimage.transform import resize
from pydicom.filereader import dcmread
import h5py
import multiprocessing


# linear attenuations in m^-1
MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

# path to LIDC-IDRI dataset
DATA_PATH = '/localdata/LIDC-IDRI'
# path to output
PATH = '/localdata/lodopab'

FILE_LIST_FILE = os.path.join(os.path.dirname(__file__),
                              'lidc_idri_file_list.json')

os.makedirs(PATH, exist_ok=True)

with open(FILE_LIST_FILE, 'r') as f:
    file_list = json.load(f)

# ~26cm x 26cm images
MIN_PT = [-0.13, -0.13]
MAX_PT = [0.13, 0.13]


def lidc_idri_gen(part='train'):
    seed = 0
    if part == 'validation':
        seed = 1
    elif part == 'test':
        seed = 2
    r = np.random.RandomState(seed)
    for dcm_file in file_list[part]:
        dataset = dcmread(os.path.join(DATA_PATH, dcm_file))

        # crop to largest rectangle in centered circle
        array = dataset.pixel_array[75:-75, 75:-75].astype(np.float32).T

        # rescale by dicom meta info
        array *= dataset.RescaleSlope
        array += dataset.RescaleIntercept

        # add noise to get continuous values from discrete ones
        array += r.uniform(0., 1., size=array.shape)

        # convert values
        array *= (MU_WATER - MU_AIR) / 1000
        array += MU_WATER
        array /= MU_MAX
        np.clip(array, 0., 1., out=array)

        yield array


lidc_idri_gen_len = {p: len(file_list[p]) for p in
                     ['train', 'validation', 'test']}

NUM_ANGLES = 1000
RECO_IM_SHAPE = (362, 362)

# image shape for simulation
IM_SHAPE = (1000, 1000)  # images will be scaled up from (362, 362)

reco_space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT,
                               shape=RECO_IM_SHAPE, dtype=np.float32)
space = odl.uniform_discr(min_pt=MIN_PT, max_pt=MAX_PT, shape=IM_SHAPE,
                          dtype=np.float64)

reco_geometry = odl.tomo.parallel_beam_geometry(
    reco_space, num_angles=NUM_ANGLES)
geometry = odl.tomo.parallel_beam_geometry(
    space, num_angles=NUM_ANGLES, det_shape=reco_geometry.detector.shape)

IMPL = 'astra_cpu'
reco_ray_trafo = odl.tomo.RayTransform(reco_space, reco_geometry, impl=IMPL)
ray_trafo = odl.tomo.RayTransform(space, geometry, impl=IMPL)

PHOTONS_PER_PIXEL = 8192

rs = np.random.RandomState(3)

NUM_SAMPLES_PER_FILE = 128


def forward_fun(im):
    # upsample ground_truth from 362px to 1000px in each dimension
    # before application of forward operator in order to avoid
    # inverse crime
    im_resized = resize(im * MU_MAX, IM_SHAPE, order=1)

    # apply forward operator
    data = ray_trafo(im_resized).asarray()

    data *= (-1)
    np.exp(data, out=data)
    data *= PHOTONS_PER_PIXEL
    return data


for part in ['train', 'validation', 'test']:
    gen = lidc_idri_gen(part)
    n_files = ceil(lidc_idri_gen_len[part] / NUM_SAMPLES_PER_FILE)
    for filenumber in tqdm(range(n_files), desc=part):
        obs_filename = os.path.join(
            PATH, 'observation_{}_{:03d}.hdf5'.format(part, filenumber))
        ground_truth_filename = os.path.join(
            PATH, 'ground_truth_{}_{:03d}.hdf5'.format(part, filenumber))
        with h5py.File(obs_filename, 'w') as observation_file,\
                h5py.File(ground_truth_filename, 'w') as ground_truth_file:
            observation_dataset = observation_file.create_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + ray_trafo.range.shape,
                dtype=np.float32, chunks=True)
            ground_truth_dataset = ground_truth_file.create_dataset(
                'data', shape=(NUM_SAMPLES_PER_FILE,) + reco_space.shape,
                maxshape=(NUM_SAMPLES_PER_FILE,) + reco_space.shape,
                dtype=np.float32, chunks=True)
            im_buf = [im for im in islice(gen, NUM_SAMPLES_PER_FILE)]
            with multiprocessing.Pool(20) as pool:
                data_buf = pool.map(forward_fun, im_buf)

            for i, (im, data) in enumerate(zip(im_buf, data_buf)):
                data = rs.poisson(data) / PHOTONS_PER_PIXEL
                np.maximum(0.1 / PHOTONS_PER_PIXEL, data, out=data)
                np.log(data, out=data)
                data /= (-MU_MAX)
                observation_dataset[i] = data
                ground_truth_dataset[i] = im

            # resize last file
            if filenumber == n_files - 1:
                observation_dataset.resize(
                    lidc_idri_gen_len[part]
                    - (n_files - 1) * NUM_SAMPLES_PER_FILE,
                    axis=0)
