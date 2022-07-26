import ocpmodels
from ocpmodels.trainers import ForcesTrainer
from ocpmodels.datasets import TrajectoryLmdbDataset
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging
setup_logging()

import numpy as np
import copy
import os
import torch

import ase
from ase import io
import os
from ocpmodels.preprocessing import AtomsToGraphs
import pickle

import lmdb
import yaml


def model_predict(test_src = "/home/litianyi/ocp/test_data",test_src_name = "/home/litianyi/ocp/test_data/0.extxyz",checkpoint_path = "/home/litianyi/ocp/checkpoints/gemnet_oc_base_s2ef_all.pt",base_yml = "/home/litianyi/ocp/configs/s2ef/all/base.yml", gemnet_yml = "/home/litianyi/ocp/configs/s2ef/all/gemnet/gemnet-oc.yml"):
    db_path = os.path.join(test_src,"test.lmdb")
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_fixed=True,
        r_distances=False,
        r_edges=True,
    )

    traj_frame = ase.io.read(test_src_name, ":")[0]


    data_object = a2g.convert(traj_frame)
    # add atom tags
    data_object.tags = torch.LongTensor(traj_frame.get_tags())
    data_object.sid = 2
    data_object.fid = 2

    txn = db.begin(write=True)
    txn.put(
        f"{0}".encode("ascii"),
        pickle.dumps(data_object, protocol=-1),
    )
    txn.commit()


    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(1, protocol=-1))
    txn.commit()

    db.sync()
    db.close()     

    with open(base_yml, "r") as f:
        config1 = yaml.load(f)

    with open(gemnet_yml, "r") as f:
        config2 = yaml.load(f)

    task = config1['task']
    model = config2['model']
    optimizer = config2['optim']
    train_src = "/home/litianyi/ocp/data/s2ef/2M/train"

    dataset = [
    {'src': train_src, 'normalize_labels': False}, # train set 
    {'src': train_src}, # val set (optional)
    {'src': test_src} # test set (optional - writes predictions to disk)
    ]

    pretrained_trainer = ForcesTrainer(
        task=task,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        identifier="gemnetoc",
        run_dir="./", # directory to save results if is_debug=False. Prediction files are saved here so be careful not to override!
        is_debug=False, # if True, do not save checkpoint, logs, or results
        #is_vis=False,
        print_every=10,
        seed=0, # random seed to use
        logger="tensorboard", # logger of choice (tensorboard and wandb supported)
        local_rank=0,
        amp=False, # use PyTorch Automatic Mixed Precision (faster training and less memory usage)
    )

    pretrained_trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    predictions = pretrained_trainer.predict(pretrained_trainer.test_loader, results_file="s2ef_results", disable_tqdm=True)

    # print(predictions["energy"])
    # print(predictions["forces"])
    res_energy = float(predictions["energy"][0])
    res_forces = predictions["forces"].tolist()[0]
    # print(type(res_energy))
    # print(type(res_forces))
    return res_energy, res_forces

# model_predict()