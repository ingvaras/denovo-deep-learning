import numpy as np
import tensorflow as tf


class CustomDataGenerator(tf.keras.preprocessing.image.ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(preprocessing_function=self.actions, **kwargs)
        self.transformations = []

    def actions(self, image):
        if np.random.random() > 0.85:
            return image
        else:
            self._check_augmentations()
            transformation = np.random.choice(self.transformations)

            return transformation(image)

    def _check_augmentations(self):

        self.transformations = []

        self.transformations.append(self._reads_cropping)
        self.transformations.append(self._reads_shuffling)
        self.transformations.append(self._nucleotides_relabeling)
        self.transformations.append(self._channels_switching)

    @staticmethod
    def _reads_cropping(img):
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))

        nreads_c = max(5, nreads_c)
        nreads_f = max(5, nreads_f)
        nreads_m = max(5, nreads_m)

        nreads_c = np.random.choice(np.arange(5, nreads_c + 1))
        nreads_f = np.random.choice(np.arange(5, nreads_f + 1))
        nreads_m = np.random.choice(np.arange(5, nreads_m + 1))

        new_img[nreads_c:, :, 0] = 0.
        new_img[nreads_f:, :, 1] = 0.
        new_img[nreads_m:, :, 2] = 0.

        return new_img

    @staticmethod
    def _nucleotides_relabeling(img):
        new_img = img.copy()

        new_ordering = list(range(4))
        np.random.shuffle(new_ordering)

        for old_idx, new_idx in enumerate(new_ordering):
            new_img[:, old_idx::4, :] = img[:, new_idx::4, :].copy()
        return new_img

    @staticmethod
    def _reads_shuffling(img):
        new_img = img.copy()

        nreads_c, nreads_f, nreads_m = tuple(np.sum(np.sum(new_img, axis=1) > 0., axis=0))

        np.random.shuffle(new_img[:nreads_c, :, 0])
        np.random.shuffle(new_img[:nreads_f, :, 1])
        np.random.shuffle(new_img[:nreads_m, :, 2])

        return new_img

    @staticmethod
    def _channels_switching(img):
        new_img = img.copy()
        new_img[:, :, 1], new_img[:, :, 2] = new_img[:, :, 2].copy(), new_img[:, :, 1].copy()

        return new_img
