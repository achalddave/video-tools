"""Show a single image and its labels from an LMDB of LabeledVideoFrames.

TODO(achald): This is a rather hacky and slow script that could be replaced
with something significantly more useful.
"""

import argparse
import random

import lmdb
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from util import video_frames_pb2

map_size = 200e9


def dump_one_lmdb(path, offset):
    with lmdb.open(path, map_size=map_size) as env, \
            env.begin().cursor() as lmdb_cursor:
        num_entries = env.stat()['entries']
        # Unfortunately, it seems the only way to set the cursor to an
        # arbitrary key index (without knowing the key) is to literally call
        # next() repeatedly.
        lmdb_cursor.next()
        for i in tqdm(range(offset)):
            lmdb_cursor.next()
        video_frame = video_frames_pb2.LabeledVideoFrame()
        video_frame.ParseFromString(lmdb_cursor.value())
        image_proto = video_frame.frame.image
        image = np.fromstring(image_proto.data,
                              dtype=np.uint8).reshape(
                                  image_proto.channels, image_proto.height,
                                  image_proto.width).transpose((1, 2, 0))
        image = Image.fromarray(image, 'RGB')
        image.save('tmp.png')
        print(lmdb_cursor.key())
        print(', '.join([label.name for label in video_frame.label]))


if __name__ == "__main__":
    # Use first line of file docstring as description if a file docstring
    # exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('lmdb')
    parser.add_argument('offset', type=int)

    args = parser.parse_args()

    dump_one_lmdb(args.lmdb, args.offset)
