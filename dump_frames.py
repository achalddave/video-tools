"""Dump video frames as images."""

import argparse
import json
import logging
import math
import os

from multiprocessing import Pool

from moviepy.editor import VideoFileClip
from moviepy import tools as mp_tools

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('video_list',
                    default=None,
                    help='File containing new-line separated paths to videos.')
parser.add_argument('output_directory',
                    default=None,
                    help='Directory to output frames to.')
parser.add_argument('--frames_per_second',
                    default=1,
                    help=('Number of frames to output per second. If 0, '
                          'dumps all frames in the clip'))

args = parser.parse_args()

logging.getLogger().setLevel(logging.INFO)

def frames_already_dumped(video_path,
                          output_directory,
                          expected_frames_per_second,
                          expected_info_path,
                          expected_name_format,
                          expected_duration):
    """Check if the output directory exists and has already been processed.

        1) Check the info.json file to see if the parameters match.
        2) Ensure that all the frames exist.

    Params:
        video_path (str)
        output_directory (str)
        expected_frames_per_second (num)
        expected_info_path (str)
        expected_name_format (str)
        expected_duration (num): Expected dration in seconds.
    """
    # Ensure that info file exists.
    if not os.path.isfile(expected_info_path):
        return False

    # Ensure that info file is valid.
    with open(expected_info_path, 'rb') as info_file:
        info = json.load(info_file)
    info_valid = info['frames_per_second'] == expected_frames_per_second \
        and info['input_video_path'] == os.path.abspath(video_path)
    if not info_valid:
        return False

    # Check that all frame paths exist.
    offset_if_one_indexed = 0
    if not os.path.exists(expected_name_format % 0):
        # If the 0th frame doesn't exist, either we haven't dumped the frames,
        # or the frames start with index 1 (this changed between versions of
        # moviepy, so we have to explicitly check). We can assume they start
        # with index 1, and continue.
        offset_if_one_indexed = 1
    expected_frame_paths = [
        expected_name_format % (i + offset_if_one_indexed)
        for i in range(int(math.floor(expected_duration *
                                      expected_frames_per_second)))
    ]
    frames_exist = all([os.path.exists(frame_path)
                        for frame_path in expected_frame_paths])
    if not frames_exist:
        return False

    # All checks passed
    return True


def dump_frames(video_path, output_directory, frames_per_second):
    """Dump frames at frames_per_second from a video to output_directory.

    If frames_per_second is None, the clip's fps attribute is used instead."""
    clip = VideoFileClip(video_path)
    info_path = '{}/info.json'.format(output_directory)
    name_format = '{}/frame%04d.png'.format(output_directory)

    if frames_per_second is None:
        frames_per_second = clip.fps
    frames_already_dumped_helper = lambda: \
            frames_already_dumped(video_path, output_directory,
                                  frames_per_second, info_path,
                                  name_format, clip.duration)
    if frames_already_dumped_helper():
        logging.info('Frames for {} exist, skipping...'.format(video_path))
        return

    clip.write_images_sequence(
        name_format.format(output_directory),
        fps=frames_per_second)
    info = {'frames_per_second': frames_per_second,
            'input_video_path': os.path.abspath(video_path)}
    with open(info_path, 'wb') as info_file:
        json.dump(info, info_file)

    if not frames_already_dumped_helper():
        logging.error("Images for {} don't seem to be dumped properly!".format(
            video_path))

def dump_frames_star(args):
    """Calls dump_frames after unpacking arguments."""
    dump_frames(*args)

def main():
    video_list = args.video_list
    output_directory = args.output_directory
    frames_per_second = float(args.frames_per_second)
    if frames_per_second == 0: frames_per_second = None
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    dump_frames_tasks = []
    with open(video_list) as f:
        for line in f:
            video_path = line.strip()
            base_filename = os.path.basename(video_path)
            output_video_directory = os.path.join(output_directory,
                                                  base_filename)
            if not os.path.isdir(output_video_directory):
                os.mkdir(output_video_directory)
            dump_frames_tasks.append((video_path, output_video_directory,
                                      frames_per_second))

    pool = Pool(8)
    try:
        pool.map(dump_frames_star, dump_frames_tasks)
    except KeyboardInterrupt:
        print 'Parent received control-c, exiting.'
        pool.terminate()


if __name__ == '__main__':
    main()
