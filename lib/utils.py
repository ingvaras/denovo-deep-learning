import math
import os


def get_steps_per_epoch(images_directory):
    total_images = len(os.listdir(os.path.join(images_directory, 'DNM'))) + len(
        os.listdir(os.path.join(images_directory, 'IV')))
    steps = math.ceil(total_images / 32)

    return steps
