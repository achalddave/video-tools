"""Split THUMOS' validation set into 'trainval' and 'valval' for MultiTHUMOS.

MultiTHUMOS annotations are not available on the training videos of THUMOS.
Instead, we split up the validation subset further into a 'train' and 'val'
subset ('trainval' and 'valval').

For each category, we pick X% (rounded up) of the videos and place them in the
'valval' set.
"""

import argparse
import logging
import random
from collections import defaultdict, OrderedDict

from util.video_tools.util.annotation import (filter_annotations_by_category,
                                              load_annotations_json)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s.%(msecs).03d: %(message)s',
                    datefmt='%H:%M:%S')


def split_constraint_fix(annotations_by_label, valtrain_videos, valval_videos):
    """Return a swap between splits to satisfy label constraints.

    If each label doesn't have one video in valtrain and valval, return a
    swapping of videos that will satisfy one constraint.

    Returns:
        swap_pair (tuple or None): Contains one video from valtrain and one
            from valval. If None, then all constraints are satisfied.
    """
    valtrain_videos = set(valtrain_videos)
    valval_videos = set(valval_videos)
    swap_pair = None
    # We attempt to satisfy constraints for the labels with more annotations
    # first. Satisfying constraints for labels with fewer annotations should
    # be less likely to violate an earlier constraint.
    label_annotations_length_sorted = sorted(annotations_by_label.items(),
                                             key=lambda (_, v): len(v),
                                             reverse=True)
    for label, annotations in label_annotations_length_sorted:
        filenames = set(x.filename for x in annotations)
        if len(filenames) < 2:
            raise Exception('Label %s has only one file!' % label)
        if not filenames.intersection(valtrain_videos):
            # Swap one file for this label from the validation set with a
            # random file from the train set.
            swap_pair = (random.choice(tuple(valtrain_videos)),
                         random.choice(tuple(filenames)))
            break
        elif not filenames.intersection(valval_videos):
            # Swap one file for this label from the train set with a
            # random file from the validation set.
            swap_pair = (random.choice(tuple(filenames)),
                         random.choice(tuple(valval_videos)))
            break
    if swap_pair is not None:
        logging.info('Swapping (%s, %s) for label %s', swap_pair[0],
                     swap_pair[1], label)
    return swap_pair


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('val_annotations_json')
    parser.add_argument('output_trainval_names',
                        help='File to output trainval video names to.')
    parser.add_argument('output_valval_names',
                        help='File to output valval video names to.')
    parser.add_argument(
        '--val_portion',
        default=0.2,
        type=float,
        help='Portion of videos to place in the val set.')
    parser.add_argument('--seed', default=0, type=int)

    args = parser.parse_args()
    random.seed(args.seed)

    val_annotations_unordered = load_annotations_json(
        args.val_annotations_json)
    filenames = sorted(val_annotations_unordered.keys())
    # Create a fixed ordering for val_annotations so all runs have the same
    # ordering when looping over the dictionary.
    val_annotations = OrderedDict()
    val_annotations = {filename: val_annotations_unordered[filename]
                       for filename in filenames}

    num_valval = int(round(args.val_portion * len(filenames)))
    num_valtrain = len(filenames) - num_valval
    random.shuffle(filenames)
    valval_videos = set(filenames[:num_valval])
    valtrain_videos = set(filenames[num_valval:])

    annotations_by_label = OrderedDict()
    for file_annotations in val_annotations.values():
        for annotation in file_annotations:
            if annotation.category not in annotations_by_label:
                annotations_by_label[annotation.category] = []
            annotations_by_label[annotation.category].append(annotation)
    for label, annotations in annotations_by_label.items():
        logging.info('%s: %s files', label,
                     len(set([x.filename for x in annotations])))

    while True:
        swap = split_constraint_fix(annotations_by_label, valtrain_videos,
                                    valval_videos)
        if swap is None:
            break
        valtrain_videos.remove(swap[0])
        valtrain_videos.add(swap[1])

        valval_videos.remove(swap[1])
        valval_videos.add(swap[0])

    logging.info('# train videos: %s', len(valtrain_videos))
    logging.info('# val videos: %s', len(valval_videos))
    with open(args.output_trainval_names, 'wb') as train_f, \
            open(args.output_valval_names, 'wb') as val_f:
        train_f.writelines([x + '\n' for x in valtrain_videos])
        val_f.writelines([x + '\n' for x in valval_videos])

if __name__ == "__main__":
    main()
