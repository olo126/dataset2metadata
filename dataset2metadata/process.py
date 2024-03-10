import os
import traceback
import pathlib
from importlib.machinery import SourceFileLoader
from threading import Thread
from queue import Empty, Queue
from typing import List
from tqdm import tqdm
import time
import json

import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hashlib
import logging
from pathlib import Path

import fsspec
import yaml
from PIL import ImageFile
import wandb

from dataset2metadata.dataloaders import create_loader
from dataset2metadata.registry import update_registry
from dataset2metadata.utils import topsort, download_all
from dataset2metadata.writer import Writer

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.getLogger().setLevel(logging.INFO)


def check_yml(yml):
    # manditory fields in the yml
    yml_fields = [
        "models",
        "nworkers",
        "batch_size",
        "device",
        "input_tars",
        "output_metadata_dir",
    ]

    yml_optional_fields = [
        "postprocess_columns",
        "postprocess_features",
        "additional_fields",
        "custom_pypath",
        "reprocess",
        "use_datacomp_keys",
        "tars_per_wds",  # TODO: support braceexpand
        "logging",
    ]

    for f in yml_fields:
        if f not in yml:
            raise ValueError(f"missing required yml field: {f}")

    for f in yml:
        if f not in yml_fields + yml_optional_fields:
            raise ValueError(f"unknown field: {f}")


def process_helper(
    yml,
    model_lookup,
    postprocess_feature_lookup,
    postprocess_parquet_lookup,
    in_queue: Queue,
    out_queue: Queue,
):
    # initializing models
    models = {m_str: model_lookup[m_str](yml["device"]) for m_str in yml["models"]}

    # deciding order to run them in based on dependencies
    topsort_order = topsort(
        {m_str: model_lookup[m_str].dependencies for m_str in yml["models"]}
    )

    logging.info(f"topsort model evaluation order: {topsort_order}")

    total_start_time = time.time()
    image_count = 0
    while True:
        # for name, group in zip(names, groups):
        group, name = None, None
        try:
            group, name = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        process_time_start = time.time()

        # create dataloader based on user input
        # CHANGED yml['additional_field'] to None

        dataloader, input_map = create_loader(
            group,
            yml["models"],
            None,
            yml["nworkers"],
            yml["batch_size"],
        )

        # initialize the writer that stores results and dumps them to store
        feature_fields = []
        parquet_fields = []
        if "postprocess_features" in yml:
            feature_fields = yml["postprocess_features"]
        if "postprocess_columns" in yml:
            parquet_fields.extend(yml["postprocess_columns"])
        #if "additional_fields" in yml:
        #    parquet_fields.extend(yml["additional_fields"])

        writer = Writer(
            name,
            feature_fields,
            parquet_fields,
            yml["use_datacomp_keys"] if "use_datacomp_keys" in yml else False,
        )

        if dataloader == None:
            logging.info(f"starting to write: {writer.name}")
            logging.info(f"input not found: {group}")
            out_queue.put_nowait((writer, yml["output_metadata_dir"]))
            time.sleep(5)
            logging.info(f"continuing on gpu!")
            continue

        t_loader_end = None
        t_inference_start = None
        t_inference_end = None
        t_loader_start = time.time()

        for sample in dataloader:
            image_count += sample[0].size()[0]
            t_loader_end = time.time()

            model_outputs = {}

            t_inference_start = time.time()
            # eval all models sequentially in a top sort order
            for m_str in topsort_order:
                model_input = []
                cache = {}

                # fill the model input
                for i in input_map[m_str]:
                    if isinstance(i, int):
                        if models[m_str].to_device and i not in cache:
                            if isinstance(sample[i], List):
                                # if list needs to be moved to device transpose and move it
                                sample[i] = list(zip(*sample[i]))
                                for j in range(len(sample[i])):
                                    sample[i][j] = torch.cat(sample[i][j]).to(
                                        yml["device"]
                                    )
                                cache[i] = sample[i]
                            else:
                                cache[i] = sample[i].to(yml["device"])
                        else:
                            cache[i] = sample[i]

                        model_input.append(cache[i])
                    else:
                        # use previously computed outputs and new inputs
                        # NOTE: assume downstream model consumes on same device as upstream
                        assert i in model_outputs
                        model_input.append(model_outputs[i])

                with torch.no_grad():
                    model_outputs[m_str] = models[m_str](*model_input)

                # TODO: make this more general, right now assumes last entry is json fields
                #if len(yml["additional_fields"]):
                #    model_outputs["json"] = sample[-1]

            t_inference_end = time.time()

            if (model_outputs['nsfw-image-oai-clip-vit-l-14'].size() == torch.Size([])):
                model_outputs['nsfw-image-oai-clip-vit-l-14'] = model_outputs['nsfw-image-oai-clip-vit-l-14'].view(-1)


            if "postprocess_features" in yml:
                for k in yml["postprocess_features"]:
                    writer.update_feature_store(
                        k, postprocess_feature_lookup[k](model_outputs)
                    )

            if "postprocess_columns" in yml:
                for k in yml["postprocess_columns"]:
                    writer.update_parquet_store(
                        k, postprocess_parquet_lookup[k](model_outputs)
                    )

            # if additional fields from json need to be saved, add those to the store
            if "additional_fields" in yml and len(yml["additional_fields"]):
                transposed_additional_fields = postprocess_parquet_lookup[
                    "json-transpose"
                ](model_outputs)
                assert len(transposed_additional_fields) == len(
                    yml["additional_fields"]
                )
                for i, v in enumerate(transposed_additional_fields):
                    writer.update_parquet_store(yml["additional_fields"][i], v)

            sample_time = t_inference_end - t_inference_start
            loader_time = t_loader_end - t_loader_start

            writer.update_time_store(sample_time, loader_time)

            t_loader_start = time.time()
        
        entries_w_scores = []
        score_idx = 0
        all_scores = writer.parquet_store['nsfw-image-score']
        with open(group + "text_with_image_paths.jsonl", 'r') as f:
            in_file = list(f)
            for i in in_file:
                entry = json.loads(i)
                score = []
                for image in entry["images"]:
                    if image == None:
                        score.append(None)
                    else:
                        #print(all_scores[score_idx//512])
                        score.append(all_scores[score_idx//512].view(-1)[score_idx%512].item())
                        score_idx+=1
                entry['nsfw_scores'] = score
                entries_w_scores.append(entry)
        
        process_time_end = time.time()
        process_time = process_time_end - process_time_start
        writer.update_process_time(process_time)

        with open(group + "nsfw_scored_text_with_image_paths.jsonl", 'w') as f:
            for entry in entries_w_scores:
                f.write(json.dumps(entry) + "\n")

        logging.info(f"starting to write: {writer.name}")
        out_queue.put_nowait((writer, yml["output_metadata_dir"]))
        time.sleep(5)
        logging.info(f"continuing on gpu!")
    total_end_time = time.time()
    avg_im_time = image_count / (total_end_time - total_start_time)
    print(f"average image processing time: {avg_im_time}")


def write_helper(num_groups, receive_queue, done_queue):
    pbar = tqdm(total=num_groups)
    write_counter = 0

    main_exited = False

    while write_counter != num_groups:
        # keep checking for jobs finishing and update uids
        try:
            receive_queue_empty = receive_queue.empty()

            if not done_queue.empty():
                main_exited = True

            if main_exited and receive_queue_empty:
                raise ValueError(
                    f"main exited but only processed {write_counter} out of {num_groups} groups"
                )
        except TimeoutError:
            pass

        try:
            writer, out_path = receive_queue.get(timeout=10)
            logging.info("starting write")
            writer.write(out_path)
            write_counter += 1
            logging.info("ending write")
            pbar.update(1)
        except TimeoutError:
            pass
        except Empty:
            pass

    pbar.close()


def process(
    yml,
):
    job_friendly_name = "anon"
    if type(yml) is str:
        # parse yml and check resulting dict
        job_friendly_name = yml
        yml = yaml.safe_load(Path(yml).read_text())

    check_yml(yml)

    # if the user specifies specific custom implementaion of their own update the registry
    if "custom_pypath" in yml and yml["custom_pypath"] is not None:
        custom = SourceFileLoader(
            pathlib.Path(yml["custom_pypath"]).stem, yml["custom_pypath"]
        ).load_module()

        update_registry(custom)

    # import from registry here after we have updated
    # REMOVED dataset2metadata.~
    from registry import (
        model_lookup,
        postprocess_feature_lookup,
        postprocess_parquet_lookup,
    )

    # if local out dir does not exist make it
    fs, output_path = fsspec.core.url_to_fs(yml["output_metadata_dir"])
    fs.makedirs(output_path, exist_ok=True)

    # Oscar: GET subdirectories holding the data
    input_dir = yml["input_tars"] + "/" if yml["input_tars"][-1] != "/" else yml["input_tars"]
    subs = []
    for dir in os.listdir(input_dir):
        if os.path.isdir(input_dir + dir):
            subs.append(dir)
    # assign a name to the group of shards being processed
    groups = None

    # Oscar: SAVE image/jsonl files parent folder instead
    groups = [
        input_dir + sub_dir + "/text_with_images/" for sub_dir in subs
    ]
    
    names = [hashlib.md5(str(g).encode()).hexdigest() for g in groups]

    assert len(names) == len(groups)

    # cache if result already there and user does not want to reprocess
    if "reprocess" not in yml or not yml["reprocess"]:
        # cache
        # TODO: fix this to not just check of npz or parquet, but to check dynamically based on the load
        completed = fs.ls(output_path)
        completed_parquets = [p for p in completed if "npz" in p]
        completed_parquets = set([Path(s).stem for s in completed_parquets])

        filtered_names = []
        filtered_groups = []

        for i in range(len(names)):
            if names[i] in completed_parquets:
                logging.info(f"found cached result: {names[i]}")
            else:
                filtered_names.append(names[i])
                filtered_groups.append(groups[i])

        names = filtered_names
        groups = filtered_groups

    assert len(names) == len(groups)
    logging.info(f"processing {len(groups)} groups")

    if len(names) == 0:
        logging.info("all jobs already processed. exiting.")
        return

    if "logging" in yml and yml["logging"]:
        wandb.init(project="dataset2metadata", name=job_friendly_name)

    # initializing task queues
    send_queue = Queue()
    receive_queue = Queue()
    done_queue = Queue()

    for job in zip(groups, names):
        send_queue.put(job)

    # spawn thread to do the s3 writing
    p = Thread(
        target=write_helper,
        kwargs=dict(
            num_groups=len(groups),
            receive_queue=receive_queue,
            done_queue=done_queue,
        ),
    )
    p.start()

    try:
        # run processing in main thread
        process_helper(
            yml=yml,
            model_lookup=model_lookup,
            postprocess_feature_lookup=postprocess_feature_lookup,
            postprocess_parquet_lookup=postprocess_parquet_lookup,
            in_queue=send_queue,
            out_queue=receive_queue,
        )
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        print("main thread exited ungracefully")

    # signal the i/o thread it should wrap up
    done_queue.put(True)
    p.join()

    print("Done.")
