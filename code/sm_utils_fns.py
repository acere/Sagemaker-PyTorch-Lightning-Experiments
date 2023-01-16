import argparse
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)


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

    return parser


def save_model(model, model_dir):
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.state_dict(), f)
        logger.info(f"Model saved to {model_dir}")
