"""Create an LMDB mapping '<video_name>-<frame_number>' keys to label vectors.

Takes as input an HDF5 mapping '<video_name>' to a binary matrix of
(num_frames, num_labels) shape, and constructs an LMDB that has a unique key
for each frame. The labels are numpy arrays stored as byte strings, and can be
loaded calling numpy.fromstring on the values.

NOTE: The frame numbers are 1-indexed.
"""

import argparse
from StringIO import StringIO

import caffe
import h5py
import lmdb
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'labels_hdf5',
        help=('Maps video names to a binary matrix of shape (num_frames, '
              'num_labels).'))
    parser.add_argument('output_lmdb')

    args = parser.parse_args()

    map_size = 2e9

    lmdb_environment = lmdb.open(args.output_lmdb, map_size=int(map_size))
    with lmdb_environment.begin(write=True) as lmdb_transaction, h5py.File(
            args.labels_hdf5) as labels:
        for video_name, file_labels in tqdm(labels.items()):
            file_labels = np.asarray(file_labels)
            for frame_number, frame_labels in enumerate(file_labels):
                key = '{}-{}'.format(video_name, frame_number + 1)
                lmdb_transaction.put(key, frame_labels.tobytes())


if __name__ == '__main__':
    main()
