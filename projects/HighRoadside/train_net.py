#!/usr/bin/env python3
# Copyright taofuyu

"""
HighRoadside Training Script.

@train:
python /path/to/this/folder/train_net.py --config-file /path/to/configfile
@test:
TODO
@compute metrics:
TODO
"""

import os
import logging

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, launch, default_setup, DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
import highroadside_model

class HighRoadsideTrainer(DefaultTrainer):
    def __init__(self, cfg):
        logger = logging.getLogger("HighRoadside")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())







def setup(args):
    """
    init a cfg by using (default cfg) and (command lines)
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    trainer = HighRoadsideTrainer(cfg)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.config_file = '/detectron2/projects/HighRoadside/configs/HighRoadside_H85.yaml'
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

