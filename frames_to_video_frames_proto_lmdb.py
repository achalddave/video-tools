"""Create an LMDB containing VideoFrames as values.

Takes as input a root directory that contains a subdirectory for each video,
which in turn contain frames for the video. For example:

    <dataset>/
        <video_name>/
            frame1.png
            frame2.png
            ...

The only assumption is that frames are named of the form "frame[0-9]+.png".

The output LMDB contains keys "<video_name>-<frame-number>" and corresponding
VideoFrame as values. For example, video1/frame2.png is stored as the key
"video1-2".
"""

import argparse
import glob
import logging
import multiprocessing as mp
import sys

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from util import video_frames_pb2
from frames_to_caffe_datum_proto_lmdb import parse_frame_path

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')


def create_video_frame(video_name, frame_index, image_proto):
    """Create VideoFrameProto from arguments."""
    video_frame = video_frames_pb2.VideoFrame()
    video_frame.image.CopyFrom(image_proto)
    video_frame.video_name = video_name
    video_frame.frame_index = frame_index
    return video_frame


def load_image(image_path, resize_height=None, resize_width=None):
    """Load an image in video_frames.Image format.

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
    return image


def load_image_async_helper(args):
    """
    Load an image as specified by args and stores it and its path in the queue.

    The queue will be filled with (path, image) tuples.

    Args:
        args (tuple): Tuple of (queue, args for load_image)
    """
    queue = args[0]
    frame_path = args[1]
    image = load_image(*args[1:])
    queue.put((args[1], image))  # Will wait if queue is full.


def load_images_async(queue, num_processes, frame_paths, resize_height,
                      resize_width):
    """Loads images by calling load_image_datum in parallel."""
    job_arguments = [(queue, frame_path, resize_height, resize_width)
                     for frame_path in frame_paths]
    pool = mp.Pool(num_processes)
    return pool.map_async(load_image_async_helper, job_arguments)


def image_array_to_proto(image_array):
    image = video_frames_pb2.Image()
    image.channels, image.height, image.width = image_array.shape
    image.data = image_array.tostring()
    return image


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('frames_root')
    parser.add_argument('output_lmdb')
    parser.add_argument('--resize_width', default=None, nargs='?', type=int)
    parser.add_argument('--resize_height', default=None, nargs='?', type=int)
    parser.add_argument('--num_processes', default=16, nargs='?', type=int)
    args = parser.parse_args()

    # TODO(achald): Allow specifying either one, and resize the other based on
    # aspect ratio.
    if (args.resize_width is None) != (args.resize_height is None):
        raise ValueError('Both resize_width and resize_height must be '
                         'specified if either is specified.')
    map_size = int(500e9)

    batch_size = 5000

    # Load mapping from frame path to (video name, frame index)).
    frame_path_info = {
        frame_path: parse_frame_path(frame_path)
        for frame_path in glob.iglob('{}/*/*.png'.format(args.frames_root))
    }

    print 'Loaded frame paths.'

    num_paths = len(frame_path_info)
    progress = tqdm(total=num_paths)

    mp_manager = mp.Manager()
    queue = mp_manager.Queue(maxsize=batch_size)
    # Spawn threads to load images.
    load_images_async(queue, args.num_processes, frame_path_info.keys(),
                      args.resize_height, args.resize_width)

    num_stored = 0
    loaded_images = False
    while True:
        if loaded_images:
            break
        with lmdb.open(args.output_lmdb, map_size=map_size).begin(
                write=True) as lmdb_transaction:
            for _ in range(batch_size):
                num_stored += 1
                if num_stored >= num_paths:
                    loaded_images = True
                    break

                # Convert image arrays to image protocol buffers.
                frame_path, image_array = queue.get()
                image = image_array_to_proto(image_array)

                video_name, frame_index = frame_path_info[frame_path]
                video_frame_proto = create_video_frame(video_name, frame_index,
                                                       image)
                frame_key = '{}-{}'.format(video_name, frame_index)
                lmdb_transaction.put(frame_key,
                                     video_frame_proto.SerializeToString())
                progress.update(1)


if __name__ == "__main__":
    main()
