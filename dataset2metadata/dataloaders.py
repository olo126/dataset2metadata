from functools import partial
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from PIL import Image
from torchvision import transforms
from clip import clip

import webdataset as wds
from dataset2metadata.preprocessors import json_decoder
from webdataset.tariterators import (
    base_plus_ext,
    url_opener,
    tar_file_expander,
    valid_sample,
)


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    # taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # taken from: https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def get_to_tuple_directives(models, additional_fields):
    # import here as registry may have updated
    # REMOVED dataset2metadata.~
    from registry import model_lookup

    wrapper_classes = [model_lookup[m] for m in models]

    input_map = {}

    # get unique preprocessor directive, which is a raw_input, preprocessor pair
    unique_derectives = []

    for i, model_class in enumerate(wrapper_classes):
        assert len(model_class.preprocessors) == len(model_class.raw_inputs)
        preprocess_directives = [
            (model_class.raw_inputs[k], model_class.preprocessors[k])
            for k in range(len(model_class.preprocessors))
        ]
        input_map[models[i]] = []

        for j in range(len(preprocess_directives)):
            if preprocess_directives[j] not in unique_derectives:
                input_map[models[i]].append(len(unique_derectives))
                unique_derectives.append(preprocess_directives[j])
            else:
                input_map[models[i]].append(
                    unique_derectives.index(preprocess_directives[j])
                )

        if len(model_class.dependencies):
            # non-numeric, nameded dependencies, i.e., the outputs of other models
            input_map[models[i]].extend(model_class.dependencies)

    # add directives to include data from the tars into the webdataset
    if additional_fields is not None and len(additional_fields):
        # NOTE: currently no support for these additional fields being taken as inputs to models
        input_map["json"] = [
            len(unique_derectives),
        ]
        unique_derectives.append(("json", "identity"))

    return unique_derectives, input_map

class ImageData(Dataset):
    def __init__(self, input_path):
        with open(input_path + "text_with_image_paths.jsonl", 'r') as f:
            in_file = list(f)
            self.images = []
            for i in in_file:
                entry = json.loads(i)
                for image in entry["images"]:
                    if image != None:
                        self.images.append(input_path + "images/" + image)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx], 'r').convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        new_im = transform(image)
        processed_im = clip._transform(n_px=224)(Image.open(self.images[idx], 'r'))
        processed_txt = partial(clip.tokenize, truncate=True)([""])[0]
        return [processed_im, processed_txt]




#REMOVED Additional fields
def create_loader(input_shards, models, additional_fields, nworkers, batch_size):
    # import here as registry may have updated
    # REMOVED dataset2metadata.~
    from registry import preprocessor_lookup

    (
        unique_derectives,
        input_map,
    ) = get_to_tuple_directives(models, additional_fields)

    tuple_fields = [e[0] for e in unique_derectives]
    unique_preprocessors = [preprocessor_lookup[e[-1]] for e in unique_derectives]
    
    try:
        dataset = ImageData(input_shards)
    except FileNotFoundError:
        return None, input_map

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        drop_last=False,
        pin_memory=True,
    )

    return loader, input_map
