"""Create copy of an LMDB with LabeledVideoFrames values without image data.

Iterating through a LabeledVideoFrames LMDB is slow due to the image data
(bytes).  This script removes the image bytes from the LabeledVideoFrames."""

import argparse
import logging
import sys

import lmdb
from tqdm import tqdm

from util import video_frames_pb2


def write_imageless_frames_batch(read_lmdb, write_lmdb, batch_size, map_size,
                                 last_key, progress):
    """Read one batch of LabeledVideoFrames, remove images, write to output.

    Args:
        read_lmdb, write_lmdb (str): Path to LMDBs
        batch_size (int)
        map_size (int)
        last_key (str): The last key from the previous batch. This function
            will operate on the batch starting after this key. If None, starts
            from the beginning.

    Returns:
        last_key (str): The last key from this batch. If None, there are no
            more batches.
    """
    with lmdb.open(read_lmdb,
                   map_size=map_size).begin().cursor() as read_cursor, \
         lmdb.open(write_lmdb,
                   map_size=map_size).begin(write=True) as output_transaction:
        if last_key is not None:
            read_cursor.set_key(last_key)
            read_cursor.next()
        cursor_iterator = read_cursor.iternext()
        try:
            for i in range(batch_size):
                key, value = next(cursor_iterator)
                video_frame = video_frames_pb2.LabeledVideoFrame()
                video_frame.ParseFromString(value)
                video_frame.frame.image.data = ''
                output_transaction.put(key, video_frame.SerializeToString())
                progress.update(1)
        except StopIteration:
            return None
    return key


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_lmdb',
                        help='LMDB containing LabeledVideoFrames as values.')
    parser.add_argument('output_lmdb',
                        help="""Output path for LMDB with LabeledVideoFrames as
                        values without image bytes.""")
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

    with lmdb.open(args.input_lmdb, readonly=True) as env:
        num_entries = env.stat()['entries']
    batch_size = 5000
    map_size = int(500e9)
    last_key = None
    progress = tqdm(total=num_entries)
    while True:
        # logging.info('On batch %s', batch_index)
        last_key = write_imageless_frames_batch(args.input_lmdb,
                                                args.output_lmdb, batch_size,
                                                map_size, last_key, progress)
        if last_key is None:
            break


if __name__ == "__main__":
    main()
