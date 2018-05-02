"""Resize images on disk.

Takes as input a root directory that contains a subdirectory for each video,
which in turn contain frames for the video. For example:

    <dataset>/
        <video_name>/
            frame1.png
            frame2.png
            ...

The only assumption is that frames are named of the form "frame[0-9]+.png".

Outputs a directory with the same structure as the input directory, but with
resized frames.
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import re
from os import path

import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_image(image_path, resize_height, resize_width):
    """Load an image in video_frames.Image format.

    Args:
        image_path (str): Path to an image.
        resize_height (int): Height to resize an image to.
        resize_width (int): Width to resize an image to.

    Returns:
        image (PIL Image)
    """
    image = Image.open(image_path)
    image = image.resize((resize_width, resize_height))
    return image


def resize_image_async_helper(args):
    """
    Resize an image as specified by args and stores it and its path in the
    queue.

    The queue will be filled with (path, image) tuples.

    Args:
        args (tuple): Tuple of (queue, args for load_image)
    """
    queue = args[0]
    frame_path = args[1]
    image = resize_image(*args[1:])
    queue.put((args[1], image))  # Will wait if queue is full.


def resize_images_async(queue, num_processes, frame_paths, resize_height,
                        resize_width):
    """Loads images by calling load_image_datum in parallel."""
    job_arguments = [(queue, frame_path, resize_height, resize_width)
                     for frame_path in frame_paths]
    pool = mp.Pool(num_processes)
    return pool.map_async(resize_image_async_helper, job_arguments)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('frames_root')
    parser.add_argument('output_dir')
    parser.add_argument('--resize_width', required=True, type=int)
    parser.add_argument('--resize_height', required=True, type=int)
    parser.add_argument('--num_processes', default=16, nargs='?', type=int)
    parser.add_argument('--batch_write_size', default=100, nargs='?', type=int)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                        datefmt='%H:%M:%S')

    map_size = int(500e9)
    batch_size = 5000

    def output_file(frame_path):
        dirpath, filename = path.split(frame_path)
        dirname = path.split(dirpath)[1]
        return path.join(args.output_dir, dirname, filename)

    image_paths = glob.iglob('{}/*/*.png'.format(args.frames_root))
    logging.info('Globbing images, filtering resized images.')
    image_paths = [image
                   for image in tqdm(image_paths)
                   if not path.isfile(output_file(image))]

    logging.info('Resizing images')
    progress = tqdm(total=len(image_paths))
    mp_manager = mp.Manager()
    queue = mp_manager.Queue(maxsize=batch_size)
    # Spawn threads to load images.
    resize_images_async(queue, args.num_processes, image_paths,
                        args.resize_height, args.resize_width)

    num_resized = 0
    loaded_images = False
    while not loaded_images:
        frame_path, resized_image = queue.get()
        output_path = output_file(frame_path)
        output_dir = path.split(output_path)[0]

        if path.isfile(output_path):
            # This shouldn't happen since we've filtered for images that have a
            # resized copy output, but it may have been created after we
            # checked.
            logging.info('Computed %s already, skipping', output_path)
            continue

        if not path.isdir(output_dir):
            os.makedirs(output_dir)

        resized_image.save(output_path)

        num_resized += 1
        progress.update(1)
        if num_resized == len(image_paths):
            loaded_images = True
            break

if __name__ == "__main__":
    main()
