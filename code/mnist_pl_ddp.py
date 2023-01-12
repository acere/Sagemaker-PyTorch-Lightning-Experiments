import argparse
import json
import logging
import os
import sys

import boto3
import pytorch_lightning as pl
import smdistributed.dataparallel.torch.torch_smddp
import torch
from data_modules import MNISTDataModule
from lightning_fabric.plugins.environments import LightningEnvironment
from models import MNISTModel
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from sagemaker import Session
from sagemaker.experiments import load_run
from sm_experiments_lightning import SmLoadMetricsCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


region = os.getenv("AWS_REGION")
boto_session = boto3.Session(region_name=region)
sagemaker_session = Session(boto_session=boto_session)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # System Parameters
    parser.add_argument(
        "--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"])
    )
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--num-gpus", type=int, default=int(os.environ["SM_NUM_GPUS"]))
    parser.add_argument(
        "--num_nodes", type=int, default=len(json.loads(os.environ["SM_HOSTS"]))
    )

    world_size = int(os.environ["SM_NUM_GPUS"]) * len(os.environ["SM_HOSTS"])
    parser.add_argument("--world-size", type=int, default=world_size)
    parser.add_argument("--num-cpu", type=int, default=os.environ["SM_NUM_CPUS"])

    # model and output directories
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument("--epochs", type=int, default=10)

    return parser


def save_model(model, model_dir):
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.state_dict(), f)
        logger.info(f"Model saved to {model_dir}")


def main(args):
    with load_run(sagemaker_session=sagemaker_session) as run:
        logger.info("Starting data preparation")
        dm = MNISTDataModule(**vars(args), num_workers=args.num_cpu, run=run)

        logger.info("Initializing the model")
        model = MNISTModel(
            **vars(args),
        )

        logger.info("setting the environment for DDP")

        env = LightningEnvironment()
        env.world_size = lambda: int(os.environ["WORLD_SIZE"])
        env.global_rank = lambda: int(os.environ["RANK"])

        ddp = DDPStrategy(
        cluster_environment=env,  # type: ignore
        process_group_backend="smddp",
        accelerator="gpu"
        )

        trainer = pl.Trainer(
            max_epochs=int(args.epochs),
            devices=args.num_gpus,
            strategy=ddp,
            num_nodes=args.num_nodes,
            default_root_dir=args.model_dir,
            callbacks=[
                SmLoadMetricsCallback(run=run, confusion_matrix_name="confmat"),
                TQDMProgressBar(refresh_rate=100)
            ],
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = get_parser()
    parser = MNISTModel.add_model_specific_args(parser)
    parser = MNISTDataModule.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
