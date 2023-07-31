import os
from .OO3DDataset import OO3DDataset
from .data_util import ColorJitterDataset

def get_split_dataset(dataset_type, datadir, want_split="all", training=True, **kwargs):
    """
    Retrieved desired dataset class
    :param dataset_type dataset type name (srn|dvr|dvr_gen, etc)
    :param datadir root directory name for the dataset. For SRN/multi_obj data:
    if data is in dir/cars_train, dir/cars_test, ... then put dir/cars
    :param want_split root directory name for the dataset
    :param training set to False in eval scripts
    """
    dset_class, train_aug = None, None
    flags, train_aug_flags = {}, {}
    
    if dataset_type == "oo3d":
        dset_class = OO3DDataset
        flags["list_prefix"] = ""
        if training:
            flags["max_imgs"] = 100
        flags["scale_focal"] = False
        flags["z_near"] = 2.
        flags["z_far"] = 6.
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    want_train = want_split != "val" and want_split != "test" and want_split != "testsub"
    want_val = want_split != "train" and want_split != "test" and want_split != "testsub"
    want_test = want_split != "train" and want_split != "val" and want_split != "testsub"
    want_testsub = want_split != "train" and want_split != "val" and want_split != "test"

    if want_train:
        train_set = dset_class(datadir, stage="train", **flags, **kwargs)
        if train_aug is not None:
            train_set = train_aug(train_set, **train_aug_flags)

    if want_val:
        val_set = dset_class(datadir, stage="val", **flags, **kwargs)

    if want_test:
        test_set = dset_class(datadir, stage="test", **flags, **kwargs)

    if want_testsub:
        test_set = dset_class(datadir, stage="testsub", **flags, **kwargs)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test" or want_split == "testsub":
        return test_set
    return train_set, val_set, test_set
