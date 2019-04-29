"""Dump video frames as images."""

import argparse
import json
import logging
import math
import os
import subprocess
from multiprocessing import Pool

from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from tqdm import tqdm

from util.log import setup_logging


def frames_already_dumped(video_path,
                          output_directory,
                          expected_frames_per_second,
                          expected_info_path,
                          expected_name_format,
                          expected_duration,
                          log_reason=False):
    """Check if the output directory exists and has already been processed.

        1) Check the info.json file to see if the parameters match.
        2) Ensure that all the frames exist.

    Params:
        video_path (str)
        output_directory (str)
        expected_frames_per_second (num)
        expected_info_path (str)
        expected_name_format (str)
        expected_duration (num): Expected duration in seconds.
    """
    # Ensure that info file exists.
    if not os.path.isfile(expected_info_path):
        if log_reason:
            logging.info("Info path doesn't exist at %s" % expected_info_path)
        return False

    # Ensure that info file is valid.
    with open(expected_info_path, 'r') as info_file:
        info = json.load(info_file)
    info_valid = info['frames_per_second'] == expected_frames_per_second \
        and info['input_video_path'] == os.path.abspath(video_path)
    if not info_valid:
        if log_reason:
            logging.info("Info file (%s) is invalid" % expected_info_path)
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
                                      expected_frames_per_second) - 1))
    ]
    missing_frames = [x for x in expected_frame_paths if not os.path.exists(x)]
    if missing_frames:
        if log_reason:
            logging.info("Missing frames:\n%s" % ('\n'.join(missing_frames)))
        return False

    # All checks passed
    return True


def dump_frames(video_path, output_directory, frames_per_second,
                file_logger_name):
    """Dump frames at frames_per_second from a video to output_directory.

    If frames_per_second is None, the clip's fps attribute is used instead."""
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    file_logger = logging.getLogger(file_logger_name)

    try:
        video_info = ffmpeg_parse_infos(video_path)
        video_fps = video_info['video_fps']
        video_duration = video_info['duration']
    except OSError as e:
        logging.error('Unable to open video (%s), skipping.' % video_path)
        logging.exception('Exception:')
        return
    except KeyError as e:
        logging.error('Unable to extract metadata about video (%s), skipping.'
                      % video_path)
        logging.exception('Exception:')
        return
    info_path = '{}/info.json'.format(output_directory)
    name_format = '{}/frame%04d.png'.format(output_directory)

    extract_all_frames = frames_per_second is None
    if extract_all_frames:
        frames_per_second = video_fps

    frames_already_dumped_helper = lambda log_reason: \
        frames_already_dumped(video_path, output_directory,
                              frames_per_second, info_path,
                              name_format, video_duration, log_reason)

    if frames_already_dumped_helper(False):
        file_logger.info('Frames for {} exist, skipping...'.format(video_path))
        return

    successfully_wrote_images = False
    try:
        if extract_all_frames:
            cmd = ['ffmpeg', '-i', video_path, name_format]
        else:
            cmd = ['ffmpeg', '-i', video_path, '-vf',
                   'fps={}'.format(frames_per_second), name_format]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        successfully_wrote_images = True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to dump images for %s", video_path)
        logging.error(e)
        logging.error(e.output.decode('utf-8'))

    if successfully_wrote_images:
        info = {'frames_per_second': frames_per_second,
                'input_video_path': os.path.abspath(video_path)}
        with open(info_path, 'w') as info_file:
            json.dump(info, info_file)

        if not frames_already_dumped_helper(True):
            logging.error(
                "Images for {} don't seem to be dumped properly!".format(
                    video_path))


def dump_frames_star(args):
    """Calls dump_frames after unpacking arguments."""
    return dump_frames(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'video_list',
        default=None,
        help='File containing new-line separated paths to videos.')
    parser.add_argument('output_directory',
                        default=None,
                        help='Directory to output frames to.')
    parser.add_argument('--fps',
                        default=0,
                        type=float,
                        help=('Number of frames to output per second. If 0, '
                              'dumps all frames in the clip.'))
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    video_list = args.video_list
    output_directory = args.output_directory
    frames_per_second = float(args.fps)

    if frames_per_second == 0:
        frames_per_second = None

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    logging_path = args.output_directory + '/dump_frames.py'
    setup_logging(logging_path)

    dump_frames_tasks = []
    with open(video_list) as f:
        for line in f:
            video_path = line.strip()
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            output_video_directory = os.path.join(output_directory,
                                                  base_filename)
            dump_frames_tasks.append((video_path, output_video_directory,
                                      frames_per_second, logging_path))

    pool = Pool(args.num_workers)
    try:
        list(
            tqdm(
                pool.imap_unordered(dump_frames_star, dump_frames_tasks),
                total=len(dump_frames_tasks)))
    except KeyboardInterrupt:
        print('Parent received control-c, exiting.')
        pool.terminate()


if __name__ == '__main__':
    main()
