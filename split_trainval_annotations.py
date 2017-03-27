# Split trainval annotations json into train annotations and val annotations.

import argparse
import json

if __name__ == "__main__":
    # Use first line of file docstring as description.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainval_annotations', required=True)
    parser.add_argument('--train_vids_list', required=True)
    parser.add_argument('--val_vids_list', required=True)
    parser.add_argument('--train_annotations_out', required=True)
    parser.add_argument('--val_annotations_out', required=True)

    args = parser.parse_args()

    with open(args.train_vids_list) as f:
        train_vids = set([line.strip() for line in f.readlines()])
    with open(args.val_vids_list) as f:
        val_vids = set([line.strip() for line in f.readlines()])
    with open(args.trainval_annotations) as f:
        trainval_annotations = json.load(f)

    train_annotations = [annotation
                         for annotation in trainval_annotations
                         if annotation['filename'] in train_vids]
    val_annotations = [annotation
                       for annotation in trainval_annotations
                       if annotation['filename'] in val_vids]
    with open(args.train_annotations_out, 'wb') as train_out:
        json.dump(train_annotations, train_out)
    with open(args.val_annotations_out, 'wb') as val_out:
        json.dump(val_annotations, val_out)
