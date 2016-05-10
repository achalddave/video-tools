"""Create an LMDB containing Caffe Datum values from video frames on disk.

Takes as input a root directory that contains a subdirectory for each video,
which in turn contain frames for the video. For example:

    <dataset>/
        <video_name>/
            frame1.png
            frame2.png
            ...

The only assumption is that frames are named of the form "frame[0-9]+.png".

Outputs an LMDB containing keys "<video_name>-<frame-number>" and corresponding
images as values. For example, video1/frame2.png is stored as the key
"video1-2". The images are stored as Caffe's Datum protobuffer messages, in
BGR order.
"""

import argparse
import glob
import multiprocessing as mp
import re
from os import path

import caffe
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm


def frame_path_to_key(frame_path):
    """Convert an absolute frame path to a formatted frame key.

    >>> frame_path_to_key('/a/b/video/frame1.png')
    'video-1'
    >>> frame_path_to_key('/a/b/video_1234/frame01231.png')
    'video_1234-1231'
    >>> frame_path_to_key('/a/b/video_1234/frame-01231.png')  # Should be None
    >>> frame_path_to_key('/a/b/video_1234/whatever')  # Should be None
    >>> frame_path_to_key('/frame1.png')  # Should be None
    """
    dirpath, frame_filename = path.split(frame_path)
    frame_name = path.splitext(frame_filename)[0]
    video_name = path.split(dirpath)[1]
    if not video_name: return None

    frame_number = re.match('^frame([0-9]*)$', frame_name)
    if frame_number is None: return None  # No match
    frame_number = int(frame_number.group(1))

    return '{}-{}'.format(video_name, frame_number)


def load_image_datum(image_path, resize_height=None, resize_width=None):
    """Load an image in a Caffe datum in BGR order.

    Note that this method takes a tuple as an argument so it can be easily used
    as an arugment to multiprocessing.pool().map.

    Args:
        image_path (str): Path to an image.
        resize_height (int): Height to resize an image to. If 0 or None, the
            image is not resized.
        resize_width (int): Width to resize an image to. If 0 or None, the
            image is not resized.

    Returns:
        image_datum (caffe Datum): Contains the image in BGR order after
            resizing.
    """
    image = Image.open(image_path)
    if resize_height and resize_width:
        image = image.resize((resize_width, resize_height))
    # Image has shape (height, width, num_channels), where the
    # channels are in RGB order.
    image = np.array(image)
    # Convert image from RGB to BGR.
    image = image[:, :, ::-1]
    # Convert image to (num_channels, height, width) shape.
    image = image.transpose((2, 0, 1))
    return caffe.io.array_to_datum(image).SerializeToString()


def load_image_datum_helper(args):
    """Wrapper for load_image_datum for use with multiprocessing."""
    return load_image_datum(*args)

def load_image_batch(pool, frame_paths, resize_height, resize_width):
    """Loads a batch of images by calling load_image_datum in parallel."""
    job_arguments = [(frame_path, resize_height, resize_width)
                     for frame_path in frame_paths]
    return pool.map(load_image_datum_star, job_arguments)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('frames_root')
    parser.add_argument('output_lmdb')
    parser.add_argument('--resize_width', default=None, nargs='?', type=int)
    parser.add_argument('--resize_height', default=None, nargs='?', type=int)

    args = parser.parse_args()

    # TODO(achald): Allow specifying either one, and resize the other based on
    # aspect ratio.
    if (args.resize_width is None) != (args.resize_height is None):
        raise ValueError('Both resize_width and resize_height must be '
                         'specified if either is specified.')
    map_size = 500e9

    batch_size = 10000

    frame_path_key_pairs = [
        (frame_path, frame_path_to_key(frame_path))
        for frame_path in glob.iglob('{}/*/*.png'.format(args.frames_root))
    ]

    frame_path_key_pairs_batched = (
        frame_path_key_pairs[i:i + batch_size]
        for i in range(0, len(frame_path_key_pairs), batch_size)
    )
    print 'Loaded frame paths.'

    progress = tqdm(total=len(frame_path_key_pairs))
    pool = mp.Pool(8)
    for frame_path_key_pairs_batch in frame_path_key_pairs_batched:
        images_batch = load_image_batch(pool,
                                        [x[0]
                                         for x in frame_path_key_pairs_batch],
                                        args.resize_height, args.resize_height)
        lmdb_environment = lmdb.open(args.output_lmdb, map_size=int(map_size))
        with lmdb_environment.begin(write=True) as lmdb_transaction:
            for i, (frame_path,
                    frame_key) in enumerate(frame_path_key_pairs_batch):
                lmdb_transaction.put(frame_key, images_batch[i])
                progress.update(1)
        # Usually, Python garbage collects on its own just fine. In this case,
        # it seems it isn't deleting images_batch until after the next
        # images_batch is finished loading (but this is just a conjecture).
        lmdb_environment.close()
        del images_batch


if __name__ == "__main__":
    main()
