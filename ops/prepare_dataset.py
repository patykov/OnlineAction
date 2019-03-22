import os

import cv2


def extract_frames(dataDir, outputDir):
    # Output dataset directory
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    actions = os.listdir(dataDir)
    for action in actions:
        print(action)
        actionDir = os.path.join(dataDir, action)
        actionOutputDir = os.path.join(outputDir, action)

        # Action directory
        if not os.path.exists(actionOutputDir):
            os.mkdir(actionOutputDir)

        videos = os.listdir(actionDir)
        for video in videos:
            videoPath = os.path.join(actionDir, video)
            videoOutputDir = os.path.join(actionOutputDir, video.split('.')[0])

            # Video directory
            if not os.path.exists(videoOutputDir):
                os.mkdir(videoOutputDir)

            frameNum = 0
            cam = cv2.VideoCapture(videoPath)
            success, image = cam.read()
            while success:
                success, image = cam.read()
                if success:
                    cv2.imwrite(os.path.join(videoOutputDir, 'frame_{:06d}.jpg'.format(
                        frameNum)), image)
                    frameNum += 1
    print('Done!!')


def create_labels_file(dir_path, output_path):
    classes = os.listdir(dir_path)
    new_file = os.path.join(output_path, 'classes.txt')
    with open(new_file, 'w') as file:
        for i, c in enumerate(classes):
            file.write('{} {}\n'.format(i, c))


def load_labels_file(file_path):
    classes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for l in lines:
        li = l.replace('\n', '').split(' ')
        classes[li[1]] = li[0]
    return classes


def create_videos_list(dir_path, file_path, classes_file):
    file = open(file_path, 'w')
    classes_ids = load_labels_file(classes_file)
    classes = os.listdir(dir_path)

    for c in classes:
        class_path = os.path.join(dir_path, c)
        c_id = classes_ids[c]
        videos = os.listdir(class_path)
        files_text = ''.join(['{} {} {}\n'.format(
            c_id,
            os.path.join(c, v),
            len(os.listdir(os.path.join(class_path, v)))) for v in videos])
        file.write(files_text)

    file.close()
