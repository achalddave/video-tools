"""Helpers for working with annotations."""
from __future__ import division

import collections
import json

import numpy as np

Annotation = collections.namedtuple(
    'Annotation', ['filename', 'start_frame', 'end_frame', 'start_seconds',
                   'end_seconds', 'frames_per_second', 'category'])


def load_annotations_json(annotations_json_path, filter_category=None):
    """Load annotations into a dictionary mapping filenames to annotations.

    Args:
        annotations_json_path (str): Path to JSON file containing annotations.
        filter_category (str): If specified, only annotations for that category
            are returned.

    Returns:
        annotations (dict): Maps annotation file name to a list of Annotation
            objects.
    """
    with open(annotations_json_path) as f:
        annotations_list = json.load(f)
    annotations = collections.defaultdict(list)
    # Extract annotations for category
    for annotation in annotations_list:
        annotation = Annotation(**{
            fieldname: annotation[fieldname]
            for fieldname in Annotation._fields
        })
        annotations[annotation.filename].append(annotation)
    if filter_category is not None:
        annotations = filter_annotations_by_category(annotations,
                                                     filter_category)
        if not annotations:
            raise ValueError('No annotations found with category %s.' %
                             filter_category)
    return annotations


def filter_annotations_by_category(annotations, category):
    """
    Return only annotations that belong to category.

    Args:
        annotations (dict): Maps filenames to list of Annotations.
        category (str): Category to keep annotations from.

    Returns:
        filtered_annotations (dict): Maps filenames to list of Annotations.

    >>> SimpleAnnotation = collections.namedtuple(
    ...         'SimpleAnnotation', ['category'])
    >>> annotations = {'file1': [SimpleAnnotation('class1'),
    ...                          SimpleAnnotation('class2')],
    ...                'file2': [SimpleAnnotation('class2')]}
    >>> filtered = filter_annotations_by_category(annotations, 'class1')
    >>> filtered.keys()
    ['file1']
    >>> len(filtered['file1'])
    1
    """
    filtered_annotations = {}
    for filename, annotations in annotations.items():
        filtered = [x for x in annotations if x.category == category]
        if filtered:
            filtered_annotations[filename] = filtered
    return filtered_annotations


def annotations_to_frame_labels(annotations, num_frames):
    """
    Convert annotations for one category and one file to a binary label vector.

    Args:
        annotations (list of Annotation): These must correspond to exactly one
            file and one category.
        num_frames (int)

    Returns:
        frame_labels (np.array, shape (1, num_frames)): Binary vector
            indicating whether each frame is in an annotation.
    """
    if len(set([annotation.filename for annotation in annotations])) > 1:
        raise ValueError('Annotations should be for at most one filename.')
    if len(set([annotation.category for annotation in annotations])) > 1:
        raise ValueError('Annotations should be for at most one category.')

    frame_groundtruth = np.zeros((1, num_frames))
    for annotation in annotations:
        start, end = int(annotation.start_frame), int(annotation.end_frame)
        frame_groundtruth[0, start:end + 1] = 1
    return frame_groundtruth.astype(int)


def collect_frame_labels(file_annotations, frame_index, frames_per_second=None,
                         frame_step=None):
    """Collect list of labels that apply to a particular frame in a file.

    Args:
        file_annotations (list of Annotation): Annotations for a particular
            file.
        frame_index (int): Query frame index.
        frames_per_second (int): If specified, used to convert frame index to
            time in seconds.
        frame_step (int): If specified, used to convert frame index to a video
            frame. For example, if frame_step is 3, then a frame_index of 2 is
            assumed to be from frame 6 of the video. Exactly one of
            frames_per_second or frame_step must be specified.

    Returns:
        labels (list): List of label strings that apply to this frame.
    """
    assert (frames_per_second is None) != (frame_step is None), (
        "Exactly one of frames_per_second or frame_step must be specified.")
    if frames_per_second is not None:
        query_second = float(frame_index) / frames_per_second
        return sorted(list(set(annotation.category
                               for annotation in file_annotations
                               if annotation.start_seconds <= query_second <=
                               annotation.end_seconds)))
    else: # frame_step is not None
        query_frame = float(frame_index) * frame_step
        return sorted(list(set(annotation.category
                               for annotation in file_annotations
                               if annotation.start_frame <= query_frame <=
                               annotation.end_frame)))


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
            details = line.strip().split(' ')
            label_id = details[0]
            label = ' '.join(details[1:])
            label_ids[label] = int(label_id)
            if one_indexed_labels:
                label_ids[label] -= 1
    assert sorted(label_ids.values()) == range(len(label_ids)), (
        'Label ids must be consecutive and start at 0.')
    return label_ids


def in_annotation(annotation, frame_index):
    return (annotation.start_frame <= frame_index <= annotation.end_frame)


def annotations_overlap(x, y):
    x_ends_before_y_starts = x.end_frame < y.start_frame
    x_starts_after_y_ends = x.start_frame > y.end_frame
    return not (x_ends_before_y_starts or x_starts_after_y_ends)
