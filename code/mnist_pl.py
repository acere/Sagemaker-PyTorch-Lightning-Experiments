import logging
import os

import boto3
import pytorch_lightning as pl
from common_fn import get_parser, save_model
from data_modules import MNISTDataModule
from models import MNISTModel
from pytorch_lightning.callbacks import TQDMProgressBar
from sagemaker import Session
from sagemaker.experiments import load_run
from sm_experiments import SmLogger, select_loader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

region = os.getenv("AWS_REGION")
boto_session = boto3.Session(region_name=region)
sagemaker_session = Session(boto_session=boto_session)

# workaround to avoid excessive reprint of validation progress bar
progress_bar = TQDMProgressBar(refresh_rate=100)


def main(args):
    dm = MNISTDataModule(**vars(args), num_workers=args.num_cpu)
    model = MNISTModel(**vars(args))


    with select_loader()(sagemaker_session=sagemaker_session) as run:
        sm_logger = SmLogger(run)
        logger.info("Starting training with logging to SageMaker Experiments")

        trainer = pl.Trainer(
            max_epochs=int(args.epochs),
            accelerator="gpu" if args.num_gpus > 0 else "cpu",
            devices=args.num_gpus if args.num_gpus > 0 else None,
            num_nodes=args.num_nodes,
            strategy="ddp",  # ddp-spawn, the default, fails because relies on pickling
            default_root_dir=args.model_dir,
            callbacks=[progress_bar],
            logger=sm_logger,
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = get_parser()
    parser = MNISTModel.add_model_specific_args(parser)
    parser = MNISTDataModule.add_model_specific_args(parser)

    # trainer specific arguments
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    main(args)
