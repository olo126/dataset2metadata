import logging
import os
from typing import List
import json

import pandas as pd
import numpy as np
import torch
import fsspec

logging.getLogger().setLevel(logging.INFO)
from dataset2metadata.registry import d2m_to_datacomp_keys


class Writer(object):
    def __init__(
        self,
        name: str,
        feature_fields: List[str],
        parquet_fields: List[str],
        use_datacomp_keys: bool,
    ) -> None:
        self.name = name

        self.feature_store = {}
        self.parquet_store = {}

        if use_datacomp_keys:
            for e in feature_fields:
                if e in d2m_to_datacomp_keys:
                    self.feature_store[d2m_to_datacomp_keys[e]] = []
                else:
                    self.feature_store[e] = []

            for e in parquet_fields:
                if e in d2m_to_datacomp_keys:
                    self.parquet_store[d2m_to_datacomp_keys[e]] = []
                else:
                    self.parquet_store[e] = []

        else:
            # store things like CLIP features, ultimately in an npz
            self.feature_store = {e: [] for e in feature_fields}

            # store other metadata like image height, ultimately in a parquet
            self.parquet_store = {e: [] for e in parquet_fields}

        # if true, then we will remap keys using d2m_to_datacomp_keys for some fields
        self.use_datacomp_keys = use_datacomp_keys

        # store stats about how long each batch took
        self.time_store = []

        # store how long each group took to process (from reading data to writing output)
        self.total_time_store = {}

    def update_feature_store(self, k, v):
        if self.use_datacomp_keys:
            if k in d2m_to_datacomp_keys:
                self.feature_store[d2m_to_datacomp_keys[k]].append(v)
            else:
                self.feature_store[k].append(v)
        else:
            self.feature_store[k].append(v)

    def update_parquet_store(self, k, v):
        if self.use_datacomp_keys:
            if k in d2m_to_datacomp_keys:
                self.parquet_store[d2m_to_datacomp_keys[k]].append(v)
            else:
                self.parquet_store[k].append(v)
        else:
            self.parquet_store[k].append(v)

    def update_time_store(self, sample_time, loader_time):
        self.time_store.append(
            {
                "sample time (s)": sample_time,
                "loader time (s)": loader_time,
            }
        )
    
    def update_process_time(self, process_time):
        self.total_time_store["group processing time (s)"] = process_time

    def write(self, out_dir_path):
        try:
            logging.info("flattening")
            for k in self.feature_store:
                self.feature_store[k] = self._flatten_helper(
                    self.feature_store[k], to_npy=True
                )

            logging.info("more flattening")
            for k in self.parquet_store:
                self.parquet_store[k] = self._flatten_helper(self.parquet_store[k])

            num_samples = -1

            if len(self.parquet_store):
                logging.info("covert to df")
                df = pd.DataFrame.from_dict(self.parquet_store)

                num_samples = df.shape[0]

                fs, output_path = fsspec.core.url_to_fs(
                    os.path.join(out_dir_path, f"{self.name}.parquet")
                )
                with fs.open(output_path, "wb") as f:
                    logging.info("saving parquet")
                    df.to_parquet(f, engine="pyarrow")
                logging.info("file closed")
                # logging.info(f'saved metadata: {f"{self.name}.parquet"}')

            if len(self.feature_store):
                fs, output_path = fsspec.core.url_to_fs(
                    os.path.join(out_dir_path, f"{self.name}.npz")
                )
                with fs.open(output_path, "wb") as f:
                    logging.info("saving npz")
                    np.savez_compressed(f, **self.feature_store)

                logging.info(f'saved features: {f"{self.name}.npz"}')

            if len(self.time_store):
                fs, output_path = fsspec.core.url_to_fs(
                    os.path.join(out_dir_path, f"{self.name}.json")
                )
                with fs.open(output_path, "w") as f:
                    logging.info("saving json logs")

                    total_load_time = sum(
                        [e["sample time (s)"] for e in self.time_store]
                    )
                    total_inf_time = sum(
                        [e["loader time (s)"] for e in self.time_store]
                    )

                    json.dump(
                        {
                            "sample time (s)": total_load_time,
                            "loader time (s)": total_inf_time,
                            "number of samples": num_samples,
                            "group processing time (s)": self.total_time_store["group processing time (s)"],
                            "images per second": num_samples / self.total_time_store["group processing time (s)"],
                        },
                        f,
                    )

                logging.info(f'saved time logs: {f"{self.name}.json"}')

            return True

        except Exception as e:
            logging.exception(e)
            logging.error(f"failed to write metadata for shard: {self.name}")
            return False

    def _flatten_helper(self, l, to_npy=False):
        if len(l):
            if torch.is_tensor(l[0]):
                if to_npy:
                    return torch.cat(l, dim=0).float().numpy()
                return torch.cat(l, dim=0).float().tolist()
            else:
                l_flat = []
                for e in l:
                    l_flat.extend(e)

                return l_flat
        return l
