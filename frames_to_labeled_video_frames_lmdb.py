"""Create an LMDB containing LabeledVideoFrames as values.

Takes as input a root directory that contains a subdirectory for each video,
which in turn contain frames for the video. For example:

    <dataset>/
        <video_name>/
            frame1.png
            frame2.png
            ...

The only assumption is that frames are named of the form "frame[0-9]+.png".
The second input is a JSON file containing THUMOS annotations, as output by
parse_temporal_annotations.py.

The output LMDB contains keys "<video_name>-<frame-number>" and corresponding
LabeledVideoFrame as values. For example, video1/frame2.png is stored as the
key "video1-2".
"""

import argparse
import glob
import multiprocessing as mp
import logging
import sys

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from util.annotation import load_annotations_json
from frames_to_video_frames_proto_lmdb import (
    image_array_to_proto, load_images_async, parse_frame_path)
from util import video_frames_pb2


def collect_frame_labels(file_annotations, frame_index, frames_per_second):
    """Collect list of labels that apply to a particular frame in a file.

    Args:
        file_annotations (list of Annotation): Annotations for a particular
            file.
        frame_index (int): Query frame index.
        frames_per_second (int): Used to convert frame index to time in
            seconds.

    Returns:
        labels (list): List of label strings that apply to this frame.
    """
    query_second = float(frame_index) / frames_per_second
    return sorted(list(set(annotation.category
                           for annotation in file_annotations
                           if annotation.start_seconds < query_second <
                           annotation.end_seconds)))


def create_labeled_frame(video_name, frame_index, image_proto, labels,
                         label_ids):
    video_frame = video_frames_pb2.LabeledVideoFrame()
    video_frame.frame.image.CopyFrom(image_proto)
    video_frame.frame.video_name = video_name
    video_frame.frame.frame_index = frame_index
    for label in labels:
        label_proto = video_frame.label.add()
        label_proto.name = label
        label_proto.id = label_ids[label]
    return video_frame


def load_label_ids(class_mapping_path, one_indexed_labels=False):
    """
    Args:
        class_mapping_path (str): Path to a text file containing ordered list
            of labels.  Each line should be of the form "<label_id>
            <label>".
        one_indexed_labels (bool): If true, assume that <label_id>s are
            1-indexed. The output label id will be the input label id minus 1.
            If false, assume the <label_ids> are 0 indexed.

    Returns:
        label_ids (dict): Maps label name to int id. The id is equal to the
            (0-indexed) line number on which the label appeared in the class
            mapping file. Note that these are *not* the ids from THUMOS
    """
    label_ids = {}
    with open(class_mapping_path) as f:
        for i, line in enumerate(f):
            label_id, label = line.strip().split(' ')
            label_ids[label] = int(label_id)
            if one_indexed_labels:
                label_ids[label] -= 1
    assert sorted(label_ids.values()) == range(len(label_ids)), (
        'Label ids must be consecutive and start at 0.')
    return label_ids


def load_image(image_path, resize_height=None, resize_width=None):
    """Load an image in video_frames.Image format.

    Args:
        image_path (str): Path to an image.
        resize_height (int): Height to resize an image to. If 0 or None, the
            image is not resized.
        resize_width (int): Width to resize an image to. If 0 or None, the
            image is not resized.

    Returns:
        image_datum (numpy array): Contains the image in BGR order after
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


def load_image_helper(args):
    return load_image(*args)


def load_image_batch(pool, frame_paths, resize_height, resize_width):
    """Loads a batch of images by calling load_image_datum in parallel."""
    job_arguments = [(frame_path, resize_height, resize_width)
                     for frame_path in frame_paths]
    return pool.map(load_image_helper, job_arguments)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames_root', required=True)
    parser.add_argument('--annotations_json', required=True)
    parser.add_argument('--class_mapping',
                        required=True,
                        help="""
                        File containing lines of the form "<class_int_id>
                        <class_name>". The class id are assumed to be
                        0-indexed unless --one-indexed-labels is specified.""")
    parser.add_argument('--output_lmdb', required=True)

    # Optional arguments.
    parser.add_argument('--resize_width', default=None, nargs='?', type=int)
    parser.add_argument('--resize_height', default=None, nargs='?', type=int)
    parser.add_argument('--frames_per_second',
                        default=10,
                        type=float,
                        help='FPS that frames were extracted at.')
    parser.add_argument('--num_processes', default=16, nargs='?', type=int)
    parser.add_argument('--one-indexed-labels',
                        default=False,
                        action='store_true',
                        help="""If specified, the input label ids in the class
                        mapping are assumed to be 1-indexed; the output label
                        ids will be the input label id minus 1 so that they
                        are zero-indexed.""")

    args = parser.parse_args()

    logging_filepath = args.output_lmdb + '.log'
    log_formatter = logging.Formatter('%(asctime)s.%(msecs).03d: %(message)s',
                                      datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(logging_filepath)
    file_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(file_handler)

    logging.info('Writing log file to %s', logging_filepath)
    logging.info('Command line arguments: %s', sys.argv)
    logging.info('Parsed arguments: %s', args)

    # TODO(achald): Allow specifying either one, and resize the other based on
    # aspect ratio.
    if (args.resize_width is None) != (args.resize_height is None):
        raise ValueError('Both resize_width and resize_height must be '
                         'specified if either is specified.')
    map_size = int(500e9)

    batch_size = 10000

    # Load mapping from frame path to (video name, frame index)).
    frame_path_info = {
        frame_path: parse_frame_path(frame_path)
        for frame_path in glob.iglob('{}/*/*.png'.format(args.frames_root))
    }

    logging.info('Loaded frame paths.')

    annotations = load_annotations_json(args.annotations_json)

    num_paths = len(frame_path_info)
    progress = tqdm(total=num_paths)

    mp_manager = mp.Manager()
    queue = mp_manager.Queue(maxsize=batch_size)
    # Spawn threads to load images.
    load_images_async(queue, args.num_processes, frame_path_info.keys(),
                      args.resize_height, args.resize_width)
    label_ids = load_label_ids(args.class_mapping, args.one_indexed_labels)

    num_stored = 0
    loaded_images = False
    while True:
        if loaded_images:
            break
        with lmdb.open(args.output_lmdb, map_size=map_size).begin(
                write=True) as lmdb_transaction:
            # Convert image arrays to image protocol buffers.
            for _ in range(batch_size):
                # Convert image arrays to image protocol buffers.
                frame_path, image_array = queue.get()
                image = image_array_to_proto(image_array)

                video_name, frame_index = frame_path_info[frame_path]
                labels = collect_frame_labels(annotations[video_name],
                                              frame_index - 1,
                                              args.frames_per_second)
                video_frame_proto = create_labeled_frame(
                    video_name, frame_index, image, labels, label_ids)
                frame_key = '{}-{}'.format(video_name, frame_index)
                lmdb_transaction.put(frame_key,
                                     video_frame_proto.SerializeToString())
                progress.update(1)
                num_stored += 1
                if num_stored >= num_paths:
                    loaded_images = True
                    break
    logging.info('Output frames to %s.', args.output_lmdb)


if __name__ == "__main__":
    main()
