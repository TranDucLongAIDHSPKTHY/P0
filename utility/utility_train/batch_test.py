import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from utility.utility_data.data_loader import Data
from utility.utility_function.tools import mini_batch
import utility.utility_function.metrics as metrics
from config_path.config_path import checkpoint_file, result_file, temporary_file


BEST_VALIDATION_K = 20


def _serializable_metrics(result):
    if isinstance(result, list):
        return [_serializable_metrics(value) for value in result]
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in result.items()
    }


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _save_checkpoint(
    path, model, optimizer, epoch, validation_metrics, config,
    selection_metric, selection_value,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "validation_metrics": _serializable_metrics(validation_metrics),
        "selection_metric": selection_metric,
        "selection_value": selection_value,
        "configuration": dict(config),
    }
    try:
        torch.save(checkpoint, temporary)
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def general_test(dataset, model, device, config, epoch, best_results, optimizer=None, logger=None):
    """Evaluate validation, update early stopping, and persist last/best checkpoints."""
    logger = logger or logging.getLogger(__name__)
    result = Test(dataset, model, device, config, split="validation")
    output_dir = Path(dataset.training_output_dir)
    top_k = eval(config["top_K"])
    if BEST_VALIDATION_K not in top_k:
        raise ValueError(
            "top_K must include {} for best validation selection".format(BEST_VALIDATION_K)
        )
    primary_k = BEST_VALIDATION_K
    primary_index = top_k.index(primary_k)
    primary_metric = "recall@{}".format(primary_k)
    primary_value = float(result["recall"][primary_index])

    dataset.validation_history.append(
        {"epoch": epoch + 1, "primary_metric": primary_metric,
         "primary_value": primary_value, "metrics": _serializable_metrics(result)}
    )
    _write_json_atomic(result_file(output_dir, "validation"), dataset.validation_history)

    last_path = checkpoint_file(output_dir, "last")
    _save_checkpoint(
        last_path, model, optimizer, epoch + 1, result, config,
        primary_metric, primary_value,
    )
    logger.info("Saving Checkpoint: last_model=%s", last_path)

    improved = (
        best_results.get("epoch", 0) == 0
        or primary_value > float(best_results.get("primary_value", float("-inf")))
    )
    if improved:
        best_results["count"] = 0
        best_results["epoch"] = epoch + 1
        best_results["recall"] = result["recall"].copy()
        best_results["ndcg"] = result["ndcg"].copy()
        best_results["primary_metric"] = primary_metric
        best_results["primary_value"] = primary_value
        best_path = checkpoint_file(output_dir, "best_validation")
        _save_checkpoint(
            best_path, model, optimizer, epoch + 1, result, config,
            primary_metric, primary_value,
        )
        logger.info(
            "Best Validation Updated: epoch=%d recall@%d=%.8f",
            epoch + 1, primary_k, primary_value,
        )
        logger.info("Saving Checkpoint: best_validation_model=%s", best_path)
    else:
        best_results["count"] += 1
        if best_results["count"] >= int(config["early_stopping"]):
            best_results["stop"] = 99999
            logger.info(
                "Early stopping condition reached: best_epoch=%d best_recall=%s best_ndcg=%s",
                best_results["epoch"], best_results["recall"], best_results["ndcg"],
            )

    logger.info(
        "Validation Metrics: epoch=%d precision=%s recall=%s ndcg=%s",
        epoch + 1, result["precision"], result["recall"], result["ndcg"],
    )
    return result, best_results


def final_test(dataset, model, device, config, logger=None):
    """Load the best validation checkpoint and evaluate the test set exactly once."""
    logger = logger or logging.getLogger(__name__)
    output_dir = Path(dataset.training_output_dir)
    checkpoint_path = checkpoint_file(output_dir, "best_validation")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Best validation checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info("Loading Best Model: %s (epoch=%s)", checkpoint_path, checkpoint.get("epoch"))
    logger.info("Start Testing")

    if int(config.get("sparsity_test", 0)) == 1:
        result = sparsity_test(dataset, model, device, config)
    else:
        result = Test(dataset, model, device, config, split="test")
    serialized = {
        "best_validation_epoch": checkpoint.get("epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "selection_value": checkpoint.get("selection_value"),
        "best_validation_metrics": checkpoint.get("validation_metrics"),
        "test_metrics": _serializable_metrics(result),
    }
    result_path = result_file(output_dir, "final_test")
    _write_json_atomic(result_path, serialized)
    logger.info("Test Metrics: %s", serialized["test_metrics"])
    logger.info("Final test metrics saved: %s", result_path)
    return result


def Test(dataset: Data, model, device, config, split="test"):
    model = model.eval()
    topK = eval(config["top_K"])
    model_results = {
        "precision": np.zeros(len(topK)), "recall": np.zeros(len(topK)),
        "hit": np.zeros(len(topK)), "ndcg": np.zeros(len(topK)),
    }
    target_dict = dataset.validation_dict if split == "validation" else dataset.test_dict
    with torch.no_grad():
        users = list(target_dict.keys())
        if not users:
            raise ValueError("{} set contains no evaluable users".format(split))
        rating_list, ground_true_list = [], []
        batch_size = int(config["test_batch_size"])
        num_batch = (len(users) + batch_size - 1) // batch_size
        batch_count = 0
        for batch_users in mini_batch(users, batch_size=batch_size):
            batch_count += 1
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_pos_items_for_evaluation(batch_users, split)
            ground_true = [target_dict[user] for user in batch_users]
            batch_users_device = torch.Tensor(batch_users).long().to(device)
            rating = model.get_rating_for_test(batch_users_device)
            for index, items in enumerate(all_positive):
                exclude_users.extend([index] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -1
            _, rating_k = torch.topk(rating, k=max(topK))
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)
        assert num_batch == batch_count

        for single_list in zip(rating_list, ground_true_list):
            result = test_one_batch(single_list, topK)
            model_results["recall"] += result["recall"]
            model_results["precision"] += result["precision"]
            model_results["ndcg"] += result["ndcg"]
        model_results["recall"] /= float(len(users))
        model_results["precision"] /= float(len(users))
        model_results["ndcg"] /= float(len(users))
        return model_results


def test_one_batch(X, topK):
    recommender_items = X[0].numpy()
    ground_true_items = X[1]
    r = metrics.get_label(ground_true_items, recommender_items)
    precision, recall, ndcg = [], [], []
    for k_size in topK:
        recall.append(metrics.recall_at_k(r, k_size, ground_true_items))
        precision.append(metrics.precision_at_k(r, k_size, ground_true_items))
        ndcg.append(metrics.ndcg_at_k(r, k_size, ground_true_items))
    return {"recall": np.array(recall), "precision": np.array(precision), "ndcg": np.array(ndcg)}


def sparsity_test(dataset: Data, model, device, config):
    sparsity_results = []
    model = model.eval()
    topK = eval(config["top_K"])
    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {
                "precision": np.zeros(len(topK)), "recall": np.zeros(len(topK)),
                "hit": np.zeros(len(topK)), "ndcg": np.zeros(len(topK)),
            }
            rating_list, ground_true_list = [], []
            for batch_users in mini_batch(users, batch_size=int(config["test_batch_size"])):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items_for_evaluation(batch_users, "test")
                ground_true = [dataset.test_dict[user] for user in batch_users]
                batch_users_device = torch.Tensor(batch_users).long().to(device)
                rating = model.get_rating_for_test(batch_users_device)
                for index, items in enumerate(all_positive):
                    exclude_users.extend([index] * len(items))
                    exclude_items.extend(items)
                rating[exclude_users, exclude_items] = -1
                _, rating_k = torch.topk(rating, k=max(topK))
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)
            for single_list in zip(rating_list, ground_true_list):
                result = test_one_batch(single_list, topK)
                model_results["recall"] += result["recall"]
                model_results["precision"] += result["precision"]
                model_results["ndcg"] += result["ndcg"]
            model_results["recall"] /= float(len(users))
            model_results["precision"] /= float(len(users))
            model_results["ndcg"] /= float(len(users))
            sparsity_results.append(model_results)
    return sparsity_results
