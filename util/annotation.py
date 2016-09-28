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
        annotation = Annotation(**annotation)
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


def in_annotation(annotation, frame_index):
    return (annotation.start_frame <= frame_index <= annotation.end_frame)


def annotations_overlap(x, y):
    x_ends_before_y_starts = x.end_frame < y.start_frame
    x_starts_after_y_ends = x.start_frame > y.end_frame
    return not (x_ends_before_y_starts or x_starts_after_y_ends)
