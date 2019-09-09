import argparse
import os

from video_record import VideoRecord


def create_labels_file(dir_path, output_dir):
    """
    Reads a directory in which each subdir is a class name and create a file that maps each
    class name is to an integer id.
    Args:
        dir_path : Full path to the directory to be readed.
        output_dir: Full path to directory to store the 'classes.txt' file.
    Returns:
        target: Full path to the new file created.
    """
    classes = sorted(os.listdir(dir_path))
    new_file = os.path.join(output_dir, 'classes.txt')
    with open(new_file, 'w') as file:
        for i, c in enumerate(classes):
            file.write('{} {}\n'.format(i, c))
    return new_file


def load_labels_file(file_path):
    """
    Reads the 'classes.txt' file and returns a class dictionary.
    Args:
        file_path: Full path to the 'classes.txt' file.
    Returns:
        classes: Dict in the format: 'classes[class_name] = class_id'.
    """
    classes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for l in lines:
        li = l.replace('\n', '').split(' ')
        classes[li[1]] = li[0]
    return classes


def create_video_list(dir_path, output_file_path, classes_file):
    """
    Reads a directory in which each subdir is a class name that contains multiple videos and create
    a file that maps each video to its respective class id given a 'classes.txt' file.
    Args:
        dir_path : Full path to the directory to be readed.
        output_file_path: Full path to the map file to be created.
        classes_file: Full path to the 'classes.txt' file.
    """
    classes = load_labels_file(classes_file)
    files_text = ''
    for c_name, c_id in classes.items():
        class_path = os.path.join(dir_path, c_name)
        videos = sorted(os.listdir(class_path))
        print(class_path)
        usable_videos = []
        for v in videos:
            try:
                _ = VideoRecord(os.path.join(dir_path, c_name, v), None)
                usable_videos.append(v)
            except ValueError as e:
                print(e)

        files_text += ''.join(['{} {}\n'.format(
            c_id,
            os.path.join(c_name, v)) for v in usable_videos])

    with open(output_file_path, 'w') as file:
        file.write(files_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str)
    parser.add_argument('--classes_file', type=str, default=None)

    args = parser.parse_args()

    # Create classes file
    if args.classes_file is None:
        classes_output_dir = os.path.dirname(args.dir_path)  # One dir up
        classes_file = create_labels_file(args.dir_path, classes_output_dir)
    else:
        classes_file = args.classes_file

    # Create clips list file
    list_file = os.path.join(os.path.dirname(args.dir_path),
                             os.path.basename(args.dir_path) + '_list.txt')
    create_video_list(args.dir_path, list_file, classes_file)
