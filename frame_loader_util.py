import multiprocessing as mp
import numpy as np
import re
from os import path

from PIL import Image


def load_image(image_path, resize_height=None, resize_width=None):
    """Load an image in video_frames.Image format.

    Args:
        image_path (str): Path to an image.
        resize_height (int): Height to resize an image to. If 0 or None, the
            image is not resized.
        resize_width (int): Width to resize an image to. If 0 or None, the
            image is not resized.

    Returns:
        image (numpy array): Contains the image in BGR order after resizing.
    """
    image_pil = Image.open(image_path)
    if resize_height and resize_width:
        image_pil = image_pil.resize((resize_width, resize_height))
    # Image has shape (height, width, num_channels), where the
    # channels are in RGB order.
    image = np.asarray(image_pil)
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


def parse_frame_path(frame_path, frame_prefix='frame'):
    """Convert an absolute frame path to a (video name, frame number) tuple.

    >>> parse_frame_path('/a/b/video/frame1.png')
    ('video', 1)
    """
    dirpath, frame_filename = path.split(frame_path)
    frame_name = path.splitext(frame_filename)[0]
    video_name = path.split(dirpath)[1]
    if not video_name: return None

    frame_number = re.match('^{}([0-9]*)$'.format(frame_prefix), frame_name)
    if frame_number is None: return None  # No match
    frame_number = int(frame_number.group(1))

    return (video_name, frame_number)


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
    frame_info = parse_frame_path(frame_path)
    if frame_info is None: return frame_info
    return '{}-{}'.format(*frame_info)
