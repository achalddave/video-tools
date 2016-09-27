"""Label a video with an annotation, clip it around the annotation."""

import argparse
from os import path

from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def label_clip(video_path, label, start_second, end_second):
    clip = VideoFileClip(video_path)
    text_clip = TextClip(label, fontsize=40, color='white', bg_color='red')
    text_clip = text_clip.set_pos(('center', 'bottom'))
    text_clip = text_clip.set_start(start_second).set_duration(end_second -
                                                               start_second)
    return CompositeVideoClip([clip, text_clip])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video_file', type=str)
    parser.add_argument('label', type=str)
    parser.add_argument('start_second', type=float)
    parser.add_argument('end_second', type=float)
    parser.add_argument('output_prefix', type=str)
    parser.add_argument(
        'context_seconds',
        nargs='?',
        default=1.,
        type=float,
        help='Seconds of context to pad before and after the label.')

    args = parser.parse_args()

    labeled_clip = label_clip(args.video_file, args.label, args.start_second,
                              args.end_second)
    padded_start = args.start_second - args.context_seconds
    padded_end = args.end_second + args.context_seconds
    labeled_clip = labeled_clip.subclip(padded_start, padded_end)

    base_videoname = path.splitext(path.basename(args.video_file))[0]
    output_file = path.join(args.output_prefix, '{}-{}-{}-{}.mp4'.format(
        base_videoname, args.label, args.start_second, args.end_second))
    labeled_clip.write_videofile(output_file)
