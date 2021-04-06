import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
import settings
from image_dataset import ImageDataset


class VISDA18Dataset(ImageDataset):
    def __init__(self, img_size, range01, rgb_order,
                 file_list_path, images_dir, has_ground_truth, dummy=False):

        names = []
        paths = []
        y = None
        class_names = None
        n_classes = 13
        if has_ground_truth:
            y = []
            with open(file_list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() != '':
                        name, _, cls_i = line.rpartition(' ')
                        cls_i = int(cls_i)
                        names.append(name)
                        paths.append(os.path.join(images_dir, name))
                        y.append(cls_i)
            class_names = [''] * n_classes
            y = np.array(y, dtype=np.int32)
            for cls_i in range(n_classes):
                first = np.arange(y.shape[0])[y == cls_i][0]
                class_names[cls_i] = names[int(first)].partition('/')[0]
        else:
            y = None
            with open(file_list_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    name = line.strip()
                    if name != '':
                        names.append(name)
                        paths.append(os.path.join(images_dir, name))

        super(VISDA18Dataset, self).__init__(img_size, range01, rgb_order, class_names, n_classes,
                                             names, paths, y, dummy=dummy)


class TrainDataset(VISDA18Dataset):
    class ObjectImageAccessor(object):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset.samples_by_obj_id)

        def __getitem__(self, items):
            if isinstance(items, tuple):
                if len(items) == 2:
                    obj_ndx, img_ndx = items
                    if isinstance(obj_ndx, int) and isinstance(img_ndx, int):
                        sample_index = self.dataset.samples_by_obj_id[obj_ndx][img_ndx]
                        return self.dataset.load_image(self.dataset.paths[sample_index])
                    elif isinstance(obj_ndx, np.ndarray) and isinstance(img_ndx, np.ndarray):
                        xs = []
                        for i, j in zip(obj_ndx, img_ndx):
                            sample_index = self.dataset.samples_by_obj_id[i][j]
                            img = self.dataset.load_image(self.dataset.paths[sample_index])
                            xs.append(img)
                        return xs
                    else:
                        raise TypeError('items should be (int, int), (long, long) or '
                                        '(ndarray, ndarray), not {}'.format((type(obj_ndx), type(img_ndx))))
                else:
                    raise TypeError('items should have length 2, not {}'.format(
                        len(items)
                    ))
            else:
                raise TypeError('items should be a tuple, not a {}'.format(
                    type(items)
                ))

    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        train_dir = settings.get_data_dir('visda18_clf_train')
        file_list_path = os.path.join(train_dir, 'image_list_red.txt')
        super(TrainDataset, self).__init__(img_size, range01, rgb_order,
                                           file_list_path, train_dir, has_ground_truth=True,
                                           dummy=dummy)

        self.object_ids = []
        self.obj_id_to_idx = {}
        for sample_idx, name in enumerate(self.names):
            object_id, _ = os.path.splitext(name)
            obj_id_idx = self.obj_id_to_idx.setdefault(object_id, len(self.obj_id_to_idx))
            self.object_ids.append(obj_id_idx)
        self.object_ids = np.array(self.object_ids, dtype=np.int32)
        sample_ndxs = np.arange(len(self.object_ids))
        self.samples_by_obj_id = [sample_ndxs[self.object_ids == i] for i in range(len(self.obj_id_to_idx))]
        self.obj_X = self.ObjectImageAccessor(self)


class ValidationDataset(VISDA18Dataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        val_dir = settings.get_data_dir('visda18_clf_validation')
        file_list_path = os.path.join(val_dir, 'image_list_red.txt')
        super(ValidationDataset, self).__init__(img_size, range01, rgb_order,
                                                file_list_path, val_dir, has_ground_truth=True,
                                                dummy=dummy)


class TestDataset(VISDA18Dataset):
    def __init__(self, img_size, range01=False, rgb_order=False, dummy=False):
        test_dir = settings.get_data_dir('visda18_clf_test')
        file_list_path = os.path.join(test_dir, 'image_list_red.txt')
        super(TestDataset, self).__init__(img_size, range01, rgb_order,
                                          file_list_path, test_dir, has_ground_truth=False,
                                          dummy=dummy)


if __name__ == '__main__':
    val = ValidationDataset((96, 96), None, None)
    print(val.class_names)
    print(val.y.shape, val.y.min(), val.y.max())
    print(len(val.paths))

    train = TrainDataset((96, 96), None)
    print(train.y.shape, train.y.min(), train.y.max())
    print(len(train.paths))
    print('***')
