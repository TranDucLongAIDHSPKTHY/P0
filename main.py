"""PyTorch implementation of ID-based graph recommender systems."""

__author__ = "Yi Zhang"

import importlib
import logging
import os
import sys
from time import time

import torch

import Parser
import utility.utility_data.data_loader as data_loader
import utility.utility_function.tools as tools
import utility.utility_train.batch_test as batch_test
from config_path.config_path import (
    model_config_file,
    model_result_dir,
    training_log_file,
    verified_dataset_dir,
)


MODEL_LIST = {
    "0": "unknown", "1": "MFBPR", "2": "GCMC", "3": "GCCF", "4": "NGCF",
    "5": "LightGCN", "6": "IMPGCN", "7": "SGL", "8": "CVGA", "9": "SimGCL",
    "10": "XSimGCL", "11": "DirectAU", "12": "NCL", "13": "HCCF", "14": "LightGCL",
    "15": "DCCF", "16": "CGCL", "17": "MAWU", "18": "RecDCL", "19": "BIGCF",
    "20": "SCCF", "21": "EGCF", "22": "LightGODE", "23": "LightGCN_pp",
    "24": "MixRec", "25": "LightCCF", "26": "LightCSCF",
}


def configure_training_logging(model_name=None, dataset_name=None, seed=None):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    log_path = None
    if model_name is not None and dataset_name is not None:
        log_path = training_log_file(model_name, dataset_name, seed)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logfile = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        logfile.setFormatter(formatter)
        root_logger.addHandler(logfile)
    return logging.getLogger("training"), log_path


def select_model(args, logger):
    if args.model != "unknown":
        return args.model
    logger.info("Available models: %s", ", ".join(MODEL_LIST.values()))
    while True:
        selected_num = input("Please input the identifier of the model:")
        if selected_num in MODEL_LIST and selected_num != "0":
            return MODEL_LIST[selected_num]
        logger.error("Input Error. Please select a valid implemented model identifier.")


def run_training_seed(model_name, args, config, default_config, device, seed):
    started_at = time()
    dataset = None
    args.seed = seed
    logger, log_path = configure_training_logging(model_name, config["dataset"], seed)
    logger.info(
        "Start Training: model=%s dataset=%s seed=%d",
        model_name,
        config["dataset"],
        seed,
    )
    logger.info("Training log: %s", log_path)
    try:
        for key in config:
            if config[key] != default_config[key]:
                logger.info(
                    "CLI configuration override: %s=%s (default=%s)",
                    key,
                    config[key],
                    default_config[key],
                )

        if args.seed_flag:
            tools.set_seed(seed)
        logger.info("Random seed: %d (enabled=%s)", seed, args.seed_flag)

        dataset_path = verified_dataset_dir(config["dataset"])
        dataset = data_loader.Data(str(dataset_path), config, logger=logger)
        output_dir = model_result_dir(model_name, config["dataset"], seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset.training_output_dir = output_dir

        logger.info("Dataset Loaded: %s", dataset_path)
        logger.info("Train Dataset Size: %d", dataset.num_train)
        logger.info("Validation Dataset Size: %d", dataset.num_validation)
        logger.info("Test Dataset Size: %d", dataset.num_test)
        logger.info(dataset.get_statistics())

        trainer_class = getattr(importlib.import_module("models." + model_name), "Trainer")
        recommender = trainer_class(args, config, dataset, device, logger)
        logger.info("Model Initialized: %s on %s", model_name, device)
        for key, value in config.items():
            logger.info("Configuration %s=%s", key, value)

        recommender.train()
        logger.info("Loading Best Model")
        batch_test.final_test(dataset, recommender.model, device, config, logger)
        logger.info("Training Finished")
        return 0
    except Exception:
        epoch = getattr(dataset, "current_epoch", "not started") if dataset is not None else "not started"
        logger.exception("Training failed; seed=%d epoch=%s", seed, epoch)
        return 1
    finally:
        logger.info("Seed %d Training Time: %.3f seconds", seed, time() - started_at)


def main():
    started_at = time()
    logger, _ = configure_training_logging()
    try:
        preliminary_args = Parser.parse_model_args()
        model_name = select_model(preliminary_args, logger)
        config_path = model_config_file(model_name)
        default_config = tools.read_configuration(str(config_path), model_name)
        args = Parser.parse_args(default_config)
        args.model = model_name
        config = Parser.apply_config_overrides(default_config, args)

        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seeds = [args.seed] if args.seed is not None else list(args.seeds)
        seeds = list(dict.fromkeys(seeds))
        logger.info("Training seeds: %s", seeds)

        failed_seeds = []
        for seed in seeds:
            result = run_training_seed(
                model_name,
                args,
                config,
                default_config,
                device,
                seed,
            )
            if result != 0:
                failed_seeds.append(seed)

        logger, _ = configure_training_logging()
        if failed_seeds:
            logger.error("Training failed for seeds: %s", failed_seeds)
            return 1
        logger.info("Training completed for seeds: %s", seeds)
        return 0
    except Exception:
        logger.exception("Training setup failed")
        return 1
    finally:
        logger.info("Total Multi-seed Training Time: %.3f seconds", time() - started_at)

if __name__ == "__main__":
    sys.exit(main())
