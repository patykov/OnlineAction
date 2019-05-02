import argparse
import os

import cv2


def extract_frames(dir_path, output_path):
    # Output dataset directory
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    actions = sorted(os.listdir(dir_path))
    for action in actions:
        print(action)
        action_dir = os.path.join(dir_path, action)
        action_output_dir = os.path.join(output_path, action)

        # Action directory
        if not os.path.exists(action_output_dir):
            os.mkdir(action_output_dir)

        videos = os.listdir(action_dir)
        for video in videos:
            video_path = os.path.join(action_dir, video)
            video_output_dir = os.path.join(action_output_dir, video.split('.')[0])

            # Video directory
            if not os.path.exists(video_output_dir):
                os.mkdir(video_output_dir)

            frameNum = 0
            cam = cv2.VideoCapture(video_path)
            success, image = cam.read()
            while success:
                success, image = cam.read()
                if success:
                    cv2.imwrite(os.path.join(video_output_dir, 'frame_{:06d}.jpg'.format(
                        frameNum)), image)
                    frameNum += 1
    print('Done!!')


def create_labels_file(dir_path, output_path):
    classes = sorted(os.listdir(dir_path))
    new_file = os.path.join(output_path, 'classes.txt')
    with open(new_file, 'w') as file:
        for i, c in enumerate(classes):
            file.write('{} {}\n'.format(i, c))
    return new_file


def load_labels_file(file_path):
    classes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for l in lines:
        li = l.replace('\n', '').split(' ')
        classes[li[1]] = li[0]
    return classes


def create_videos_list(dir_path, file_path, classes_file):
    classes = load_labels_file(classes_file)
    files_text = ''
    for c_name, c_id in classes.items():
        class_path = os.path.join(dir_path, c_name)
        videos = os.listdir(class_path)
        files_text += ''.join(['{} {} {}\n'.format(
            c_id,
            os.path.join(c_name, v),
            len(os.listdir(os.path.join(class_path, v)))) for v in videos])

    with open(file_path, 'w') as file:
        file.write(files_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--classes_file', type=str, default=None)

    args = parser.parse_args()

    # Create classes file
    if args.classes_file is None:
        classes_output_dir = os.path.dirname(args.dir_path)  # One dir up
        classes_file = create_labels_file(args.dir_path, classes_output_dir)
    else:
        classes_file = args.classes_file

    # Extract frames
    extract_frames(args.dir_path, args.output_path)

    # Create list file
    list_file = os.path.splitext(args.dir_path)[1] + '_list.txt'
    create_videos_list(args.output_path, list_file, classes_file)
