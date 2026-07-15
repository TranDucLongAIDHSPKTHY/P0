"""Hậu xử lý dữ liệu sau bước merge metadata cho TaxPro-CL."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from utils import configure_logging
from config_path.config_path import (
    PROJECT_ROOT,
    dataset_verify_root_for,
    five_core_train_file,
    preprocessing_log_path,
    preprocessing_domain_sources_for,
    preprocessing_summary_file,
    preprocessed_root_for,
    split_output_file,
    split_seed_dir,
    taxonomy_distribution_file,
    temporary_file,
)


DEFAULT_SEEDS = (42, 123, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hậu xử lý dữ liệu TaxPro-CL sau merge metadata.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Thư mục gốc dự án (mặc định: thư mục cha của tools).",
    )
    parser.add_argument("--domain", choices=("all", "amazon", "yelp"), default="all")
    parser.add_argument(
        "--category-strategy",
        choices=("primary", "all"),
        default="primary",
        help="Strategy A: primary (mặc định); Strategy B: all.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
        help="Seed được xuất thành bộ dữ liệu model-ready trong dataset_verify (mặc định: 42).",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--core-size", type=int, default=5)
    parser.add_argument("--min-leaf-size", type=int, default=10)
    parser.add_argument("--coverage-threshold", type=float, default=0.80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục output (mặc định: <root>/preprocessed).",
    )
    parser.add_argument(
        "--dataset-verify-dir",
        type=Path,
        default=None,
        help="Thư mục dữ liệu model-ready (mặc định: <root>/dataset_verify).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="File log (mặc định: <root>/preprocessing_logs/preprocessing.log).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> int:
    _configure_utf8_streams()
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or preprocessed_root_for(root)).expanduser().resolve()
    dataset_verify_dir = (
        args.dataset_verify_dir or dataset_verify_root_for(root)
    ).expanduser().resolve()
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else preprocessing_log_path(root, "pipeline")
    )
    logger = configure_logging(args.log_level, log_file, "preprocessing_data")
    domains = ("amazon", "yelp") if args.domain == "all" else (args.domain,)
    logger.info("Start preprocessing")
    logger.info("Project root: %s", root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Model-ready dataset directory: %s", dataset_verify_dir)
    logger.info(
        "Configuration: domains=%s strategy=%s seeds=%s dataset_seed=%d",
        domains,
        args.category_strategy,
        args.seeds,
        args.dataset_seed,
    )
    try:
        summary = run_pipeline(
            domains=domains,
            root=root,
            output_dir=output_dir,
            dataset_verify_dir=dataset_verify_dir,
            strategy=args.category_strategy,
            seeds=args.seeds,
            dataset_seed=args.dataset_seed,
            validation_ratio=args.validation_ratio,
            core_size=args.core_size,
            min_leaf_size=args.min_leaf_size,
            coverage_threshold=args.coverage_threshold,
            logger=logger,
        )
    except Exception as error:
        logger.exception("Preprocessing failed: %s", error)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Finished preprocessing")
    return 0


def run_pipeline(
    *,
    domains: Sequence[str],
    root: Path,
    output_dir: Path,
    dataset_verify_dir: Path,
    strategy: str,
    seeds: Sequence[int],
    dataset_seed: int,
    validation_ratio: float,
    core_size: int,
    min_leaf_size: int,
    coverage_threshold: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    _validate_configuration(
        seeds, dataset_seed, validation_ratio, core_size, min_leaf_size, coverage_threshold
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_summaries = {}
    for domain in domains:
        domain_summaries[domain] = process_domain(
            domain,
            root,
            output_dir,
            dataset_verify_dir,
            strategy,
            seeds,
            dataset_seed,
            validation_ratio,
            core_size,
            min_leaf_size,
            coverage_threshold,
            logger,
        )
    overall_pass = all(value["gate_g1"]["status"] == "PASS" for value in domain_summaries.values())
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "domains": list(domains),
            "core_size": core_size,
            "min_leaf_size": min_leaf_size,
            "category_strategy": strategy,
            "validation_ratio": validation_ratio,
            "seeds": list(seeds),
            "dataset_seed": dataset_seed,
            "dataset_verify_dir": str(dataset_verify_dir),
            "coverage_threshold": coverage_threshold,
        },
        "domains": domain_summaries,
        "gate_g1": {
            "status": "PASS" if overall_pass else "FAIL",
            "failed_domains": [
                domain for domain, value in domain_summaries.items() if value["gate_g1"]["status"] == "FAIL"
            ],
        },
    }
    logger.info("Saving summary")
    write_json_atomic(preprocessing_summary_file(output_dir, "json"), summary, logger)
    write_text_atomic(
        preprocessing_summary_file(output_dir, "markdown"),
        render_summary_markdown(summary),
        logger,
    )
    logger.info("Gate G1 overall: %s", summary["gate_g1"]["status"])
    return summary


def process_domain(
    domain: str,
    root: Path,
    output_dir: Path,
    dataset_verify_dir: Path,
    strategy: str,
    seeds: Sequence[int],
    dataset_seed: int,
    validation_ratio: float,
    core_size: int,
    min_leaf_size: int,
    coverage_threshold: float,
    logger: logging.Logger,
) -> Dict[str, Any]:
    source = preprocessing_domain_sources_for(root)[domain]
    train_path = source["train"]
    test_path = source["test"]
    merged_path = source["merged"]
    item2category_path = source["item2category"]
    for path in (train_path, test_path, merged_path):
        require_file(path)

    logger.info("[%s] Loading dataset", domain)
    train_interactions = read_interactions(train_path)
    test_interactions = read_interactions(test_path)
    test_hash_before = sha256_file(test_path)

    logger.info("[%s] Applying 5-core filtering", domain)
    filtered, core_stats = iterative_core_filter(train_interactions, core_size)
    domain_dir = output_dir / domain
    five_core_path = five_core_train_file(domain_dir)
    write_interactions_atomic(five_core_path, filtered)
    logger.info(
        "[%s] 5-core filtering completed: users %d->%d, items %d->%d, interactions %d->%d, iterations=%d",
        domain,
        core_stats["users_before"],
        core_stats["users_after"],
        core_stats["items_before"],
        core_stats["items_after"],
        core_stats["interactions_before"],
        core_stats["interactions_after"],
        core_stats["iterations_to_converge"],
    )

    active_items = set(item_counts(filtered))
    logger.info("[%s] Loading merged metadata: %s", domain, merged_path)
    metadata_paths = load_taxonomy_from_merged(merged_path, active_items, logger)
    logger.info("[%s] Building taxonomy", domain)
    selected_paths = {
        item_id: choose_taxonomy_paths(metadata_paths.get(item_id, []), strategy)
        for item_id in active_items
    }
    taxonomy_before = taxonomy_statistics(selected_paths)

    logger.info("[%s] Merging small leaf categories", domain)
    final_paths, merge_stats = merge_small_leaves(selected_paths, min_leaf_size)
    taxonomy_after = taxonomy_statistics(final_paths)

    logger.info("[%s] Building item2category", domain)
    item2category = build_item2category(selected_paths, final_paths, strategy)
    write_json_atomic(item2category_path, item2category, logger)

    logger.info("[%s] Generating visualization", domain)
    visualization_path = taxonomy_distribution_file(output_dir, domain)
    generate_taxonomy_distribution(final_paths, visualization_path, domain)
    logger.info("[%s] Visualization saved: %s", domain, visualization_path)

    logger.info("[%s] Creating train/validation split", domain)
    test_pairs = interaction_pairs(test_interactions)
    split_stats = {}
    split_test_hashes: Dict[str, str] = {}
    model_ready_paths: Dict[str, str] = {}
    cold_stats = {
        "five_core_train": cold_start_groups(item_counts(filtered)),
        "test": cold_start_groups(item_counts(test_interactions)),
    }
    for seed in seeds:
        split_train, validation = split_per_user(filtered, validation_ratio, seed)
        seed_dir = split_seed_dir(domain_dir, seed)
        write_interactions_atomic(split_output_file(seed_dir, "train"), split_train)
        write_interactions_atomic(split_output_file(seed_dir, "validation"), validation)
        seed_test_path = split_output_file(seed_dir, "test")
        copy_file_atomic(test_path, seed_test_path)
        split_test_hashes[str(seed)] = sha256_file(seed_test_path)
        if seed == dataset_seed:
            model_dir = dataset_verify_dir / source["verify_name"]
            model_train_path = split_output_file(model_dir, "train")
            model_validation_path = split_output_file(model_dir, "validation")
            model_test_path = split_output_file(model_dir, "test")
            write_interactions_atomic(model_train_path, split_train)
            write_interactions_atomic(model_validation_path, validation)
            copy_file_atomic(test_path, model_test_path)
            model_ready_paths = {
                "directory": str(model_dir),
                "train": str(model_train_path),
                "validation": str(model_validation_path),
                "test": str(model_test_path),
            }
            logger.info(
                "[%s] Exported model-ready dataset seed %d: %s",
                domain,
                dataset_seed,
                model_dir,
            )
        train_pairs = interaction_pairs(split_train)
        validation_pairs = interaction_pairs(validation)
        split_stats[str(seed)] = {
            "train_users": len(split_train),
            "validation_users": len(validation),
            "train_interactions": len(train_pairs),
            "validation_interactions": len(validation_pairs),
            "test_users": len(test_interactions),
            "test_interactions": len(test_pairs),
            "test_file": str(seed_test_path),
            "test_sha256": split_test_hashes[str(seed)],
            "test_matches_source": split_test_hashes[str(seed)] == test_hash_before,
            "validation_ratio_actual": round(
                len(validation_pairs) / (len(train_pairs) + len(validation_pairs)), 8
            ),
            "leakage": {
                "train_validation_overlap": len(train_pairs & validation_pairs),
                "train_test_overlap": len(train_pairs & test_pairs),
                "validation_test_overlap": len(validation_pairs & test_pairs),
            },
            "deterministic_seed": seed,
        }
        logger.info("[%s] Computing cold-start statistics for seed %d", domain, seed)
        cold_stats["seed_{}".format(seed)] = cold_start_groups(item_counts(split_train))
        cold_stats["validation_seed_{}".format(seed)] = cold_start_groups(
            item_counts(validation)
        )

    logger.info("[%s] Running Gate G1", domain)
    test_hash_after = sha256_file(test_path)
    if not model_ready_paths:
        raise RuntimeError("Không xuất được dataset_verify cho seed {}".format(dataset_seed))
    verify_test_hash = sha256_file(Path(model_ready_paths["test"]))
    valid_items = sum(bool(paths) for paths in final_paths.values())
    coverage = valid_items / len(active_items) if active_items else 0.0
    reasons = []
    if coverage < coverage_threshold:
        reasons.append("Coverage {:.4%} thấp hơn ngưỡng {:.4%}".format(coverage, coverage_threshold))
    leaking_seeds = [
        seed
        for seed, stats in split_stats.items()
        if stats["leakage"]["train_validation_overlap"]
        or stats["leakage"]["train_test_overlap"]
        or stats["leakage"]["validation_test_overlap"]
    ]
    if leaking_seeds:
        reasons.append("Interaction leakage tại seed: {}".format(", ".join(leaking_seeds)))
    if test_hash_before != test_hash_after:
        reasons.append("test.txt đã thay đổi")
    invalid_split_test_seeds = [
        seed for seed, file_hash in split_test_hashes.items() if file_hash != test_hash_before
    ]
    if invalid_split_test_seeds:
        reasons.append(
            "Bản sao preprocessed test.txt không khớp tại seed: {}".format(
                ", ".join(invalid_split_test_seeds)
            )
        )
    if verify_test_hash != test_hash_before:
        reasons.append("dataset_verify/test.txt không giống nguyên bản")
    gate = {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons or ["Tất cả kiểm tra Gate G1 đều đạt"],
        "metadata_coverage": round(coverage, 8),
        "metadata_coverage_percent": round(coverage * 100.0, 4),
        "coverage_threshold": coverage_threshold,
        "non_leaking_protocol": {
            "all_train_validation_overlaps_zero": all(
                stats["leakage"]["train_validation_overlap"] == 0 for stats in split_stats.values()
            ),
            "all_train_test_overlaps_zero": all(
                stats["leakage"]["train_test_overlap"] == 0 for stats in split_stats.values()
            ),
            "all_validation_test_overlaps_zero": all(
                stats["leakage"]["validation_test_overlap"] == 0
                for stats in split_stats.values()
            ),
            "test_file_unchanged": test_hash_before == test_hash_after,
            "dataset_verify_test_matches_source": verify_test_hash == test_hash_before,
            "all_preprocessed_test_copies_match_source": all(
                file_hash == test_hash_before for file_hash in split_test_hashes.values()
            ),
            "test_sha256_before": test_hash_before,
            "test_sha256_after": test_hash_after,
            "metadata_uses_test_information": False,
            "taxonomy_uses_test_information": False,
            "evidence": [
                "Metadata chỉ đọc từ merged_metadata_{}.json".format(domain),
                "5-core, taxonomy frequency và leaf merge chỉ sử dụng train.txt",
                "test.txt chỉ dùng kiểm tra overlap/hash và không dùng xây taxonomy",
                "Mỗi preprocessed split chứa bản sao nguyên byte của test.txt để đánh giá",
            ],
        },
    }
    logger.info("[%s] Gate G1: %s - %s", domain, gate["status"], "; ".join(gate["reasons"]))
    logger.info("[%s] Saving output files", domain)
    return {
        "source_train": str(train_path),
        "source_test": str(test_path),
        "five_core": core_stats,
        "metadata_coverage": gate["metadata_coverage"],
        "taxonomy_statistics_before_merge": taxonomy_before,
        "taxonomy_statistics_after_merge": taxonomy_after,
        "leaf_merge_statistics": merge_stats,
        "split_statistics": split_stats,
        "model_ready_dataset": {
            "seed": dataset_seed,
            **model_ready_paths,
        },
        "cold_start_statistics": cold_stats,
        "gate_g1": gate,
        "outputs": {
            "five_core_train": str(five_core_path),
            "item2category": str(item2category_path),
            "visualization": str(visualization_path),
            "dataset_verify": model_ready_paths,
        },
    }


def read_interactions(path: Path) -> Dict[int, List[int]]:
    interactions = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            try:
                user = int(parts[0])
                items = [int(value) for value in parts[1:]]
            except ValueError as error:
                raise ValueError("Interaction lỗi tại {}:{}".format(path, line_number)) from error
            interactions.setdefault(user, []).extend(items)
    return {user: list(dict.fromkeys(items)) for user, items in interactions.items() if items}


def iterative_core_filter(
    interactions: Mapping[int, Sequence[int]], core_size: int
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    current = {user: list(items) for user, items in interactions.items() if items}
    before = interaction_statistics(current)
    iterations = 0
    while True:
        iterations += 1
        user_frequency = {user: len(items) for user, items in current.items()}
        item_frequency = item_counts(current)
        updated = {
            user: [item for item in items if item_frequency[item] >= core_size]
            for user, items in current.items()
            if user_frequency[user] >= core_size
        }
        updated = {user: items for user, items in updated.items() if items}
        if updated == current:
            break
        current = updated
    after = interaction_statistics(current)
    return current, {
        "core_size": core_size,
        "users_before": before["users"],
        "users_after": after["users"],
        "items_before": before["items"],
        "items_after": after["items"],
        "interactions_before": before["interactions"],
        "interactions_after": after["interactions"],
        "iterations_to_converge": iterations,
        "removal_rounds": max(0, iterations - 1),
    }


def interaction_statistics(interactions: Mapping[int, Sequence[int]]) -> Dict[str, int]:
    return {
        "users": len(interactions),
        "items": len(item_counts(interactions)),
        "interactions": sum(len(items) for items in interactions.values()),
    }


def item_counts(interactions: Mapping[int, Sequence[int]]) -> Counter:
    return Counter(item for items in interactions.values() for item in items)


def interaction_pairs(interactions: Mapping[int, Sequence[int]]) -> Set[Tuple[int, int]]:
    return {(user, item) for user, items in interactions.items() for item in items}


def write_interactions_atomic(path: Path, interactions: Mapping[int, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for user in sorted(interactions):
                items = sorted(set(interactions[user]))
                if items:
                    stream.write("{} {}\n".format(user, " ".join(str(item) for item in items)))
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Sao chép nguyên byte một file qua temporary file rồi thay thế nguyên tử."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(destination)
    try:
        with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
            while True:
                chunk = input_stream.read(1024 * 1024)
                if not chunk:
                    break
                output_stream.write(chunk)
        os.replace(str(temporary), str(destination))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def split_per_user(
    interactions: Mapping[int, Sequence[int]], ratio: float, seed: int
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    train, validation = {}, {}
    for user in sorted(interactions):
        items = sorted(set(interactions[user]))
        if len(items) < 2:
            train[user] = items
            continue
        validation_size = min(len(items) - 1, max(1, int(len(items) * ratio + 0.5)))
        shuffled = list(items)
        random.Random(seed * 1_000_003 + user).shuffle(shuffled)
        validation_items = set(shuffled[:validation_size])
        validation[user] = sorted(validation_items)
        train[user] = [item for item in items if item not in validation_items]
    return train, validation


def load_taxonomy_from_merged(
    path: Path, target_items: Set[int], logger: logging.Logger
) -> Dict[int, List[List[str]]]:
    result = {}
    for key, record in iter_merged_records(path, target_items):
        item_id = int(record.get("item_id", key))
        paths = []
        for raw_path in record.get("taxonomy_paths") or []:
            if isinstance(raw_path, list):
                cleaned = [str(part).strip() for part in raw_path if str(part).strip()]
                if cleaned:
                    paths.append(cleaned)
        result[item_id] = deduplicate_paths(paths)
    logger.info("Loaded taxonomy for %d/%d active items", len(result), len(target_items))
    return result


def iter_merged_records(path: Path, target_items: Set[int]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    start_pattern = re.compile(r'^  "(\d+)": \{\s*$')
    current_key: Optional[int] = None
    capture = False
    buffer: List[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if current_key is None:
                match = start_pattern.match(line.rstrip("\r\n"))
                if match:
                    current_key = int(match.group(1))
                    capture = current_key in target_items
                    buffer = ["{\n"] if capture else []
                continue
            if line.rstrip("\r\n") in ("  },", "  }"):
                if capture:
                    buffer.append("}\n")
                    yield current_key, json.loads("".join(buffer))
                current_key, capture, buffer = None, False, []
            elif capture:
                buffer.append(line[4:] if line.startswith("    ") else line)


def deduplicate_paths(paths: Iterable[Sequence[str]]) -> List[List[str]]:
    output, seen = [], set()
    for path in paths:
        key = tuple(part.casefold() for part in path)
        if key and key not in seen:
            seen.add(key)
            output.append(list(path))
    return output


def choose_taxonomy_paths(paths: Sequence[Sequence[str]], strategy: str) -> List[Tuple[str, ...]]:
    normalized = [tuple(path) for path in deduplicate_paths(paths) if path]
    if not normalized:
        return []
    if strategy == "primary":
        return [max(enumerate(normalized), key=lambda pair: (len(pair[1]), -pair[0]))[1]]
    return normalized


def taxonomy_statistics(item_paths: Mapping[int, Sequence[Tuple[str, ...]]]) -> Dict[str, Any]:
    nodes = set()
    leaf_items = defaultdict(set)
    depths, paths_per_item = [], []
    for item_id, paths in item_paths.items():
        if paths:
            paths_per_item.append(len(paths))
        for path in paths:
            depths.append(len(path))
            leaf_items[path].add(item_id)
            nodes.update(path[:depth] for depth in range(1, len(path) + 1))
    sizes = [len(items) for items in leaf_items.values()]
    largest = max(leaf_items, key=lambda key: len(leaf_items[key])) if leaf_items else None
    smallest = min(leaf_items, key=lambda key: len(leaf_items[key])) if leaf_items else None
    return {
        "total_categories": len(nodes),
        "total_leaf_categories": len(leaf_items),
        "items_with_valid_taxonomy": sum(bool(paths) for paths in item_paths.values()),
        "items_per_leaf": {" > ".join(path): len(items) for path, items in sorted(leaf_items.items())},
        "largest_leaf": {
            "taxonomy_path": list(largest) if largest else [],
            "num_items": len(leaf_items[largest]) if largest else 0,
        },
        "smallest_leaf": {
            "taxonomy_path": list(smallest) if smallest else [],
            "num_items": len(leaf_items[smallest]) if smallest else 0,
        },
        "average_leaf_size": round(sum(sizes) / len(sizes), 6) if sizes else 0.0,
        "average_taxonomy_depth": round(sum(depths) / len(depths), 6) if depths else 0.0,
        "average_taxonomy_paths": round(sum(paths_per_item) / len(paths_per_item), 6)
        if paths_per_item
        else 0.0,
    }


def merge_small_leaves(
    item_paths: Mapping[int, Sequence[Tuple[str, ...]]], threshold: int
) -> Tuple[Dict[int, List[Tuple[str, ...]]], Dict[str, Any]]:
    node_items, leaf_items = defaultdict(set), defaultdict(set)
    for item_id, paths in item_paths.items():
        for path in paths:
            leaf_items[path].add(item_id)
            for depth in range(1, len(path) + 1):
                node_items[path[:depth]].add(item_id)
    targets = {}
    for leaf in leaf_items:
        target = leaf
        while len(node_items[target]) < threshold and len(target) > 1:
            target = target[:-1]
        targets[leaf] = target
    merged, merged_items, final_leaf_items = {}, set(), defaultdict(set)
    for item_id, paths in item_paths.items():
        final, seen = [], set()
        for path in paths:
            target = targets[path]
            if target != path:
                merged_items.add(item_id)
            if target not in seen:
                seen.add(target)
                final.append(target)
                final_leaf_items[target].add(item_id)
        merged[item_id] = final
    before_sizes = [len(items) for items in leaf_items.values()]
    after_sizes = [len(items) for items in final_leaf_items.values()]
    return merged, {
        "min_leaf_size": threshold,
        "leaf_before_merge": len(leaf_items),
        "leaf_after_merge": len(final_leaf_items),
        "items_merged": len(merged_items),
        "average_leaf_size_before": round(sum(before_sizes) / len(before_sizes), 6)
        if before_sizes
        else 0.0,
        "average_leaf_size_after": round(sum(after_sizes) / len(after_sizes), 6)
        if after_sizes
        else 0.0,
    }


def build_item2category(
    original: Mapping[int, Sequence[Tuple[str, ...]]],
    final: Mapping[int, Sequence[Tuple[str, ...]]],
    strategy: str,
) -> Dict[str, Any]:
    output = {}
    for item_id in sorted(final):
        paths = list(final[item_id])
        source_paths = list(original.get(item_id, []))
        primary = paths[0] if paths else ()
        source_primary = source_paths[0] if source_paths else ()
        output[str(item_id)] = {
            "item_id": item_id,
            "leaf_category": primary[-1] if primary else None,
            "taxonomy_path": list(primary),
            "leaf_categories": [path[-1] for path in paths],
            "taxonomy_paths": [list(path) for path in paths],
            "original_leaf_category": source_primary[-1] if source_primary else None,
            "original_taxonomy_path": list(source_primary),
            "category_strategy": strategy,
        }
    return output


def generate_taxonomy_distribution(
    item_paths: Mapping[int, Sequence[Tuple[str, ...]]], path: Path, domain: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    leaf_items = defaultdict(set)
    for item_id, paths in item_paths.items():
        for taxonomy_path in paths:
            leaf_items[taxonomy_path].add(item_id)
    sizes = [len(items) for items in leaf_items.values()]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    if sizes:
        axis.hist(sizes, bins=min(60, max(10, int(len(sizes) ** 0.5))), color="#4472C4")
        if max(sizes) / max(1, min(sizes)) >= 100:
            axis.set_xscale("log")
    axis.set_title("{} taxonomy leaf distribution".format(domain.capitalize()))
    axis.set_xlabel("Number of items per leaf category")
    axis.set_ylabel("Number of leaf categories")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def cold_start_groups(counts: Mapping[int, int]) -> Dict[str, Any]:
    total = len(counts)

    def summarize(item_ids: Set[int]) -> Dict[str, Any]:
        values = [counts[item_id] for item_id in item_ids]
        return {
            "num_items": len(values),
            "percentage": round(len(values) * 100.0 / total, 4) if total else 0.0,
            "average_interactions": round(sum(values) / len(values), 6) if values else 0.0,
        }

    near = {item for item, count in counts.items() if 1 <= count <= 2}
    long_tail = {item for item, count in counts.items() if 1 <= count <= 10}
    warm = {item for item, count in counts.items() if count > 10}
    return {
        "total_observed_items": total,
        "near_cold": summarize(near),
        "long_tail": summarize(long_tail),
        "warm": summarize(warm),
        "near_cold_is_subset_of_long_tail": near <= long_tail,
    }


def write_json_atomic(path: Path, value: Any, logger: logging.Logger) -> None:
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
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def write_text_atomic(path: Path, value: str, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_file(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        os.replace(str(temporary), str(path))
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    logger.info("Saved %s (%d bytes)", path, path.stat().st_size)


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# BÁO CÁO TIỀN XỬ LÝ DỮ LIỆU",
        "",
        "- Thời điểm UTC: `{}`".format(summary["generated_at_utc"]),
        "- Gate G1 tổng thể: **{}**".format(summary["gate_g1"]["status"]),
        "",
    ]
    for domain, data in summary["domains"].items():
        core = data["five_core"]
        taxonomy = data["taxonomy_statistics_after_merge"]
        merge = data["leaf_merge_statistics"]
        gate = data["gate_g1"]
        model_ready = data["model_ready_dataset"]
        lines.extend(
            [
                "## {}".format(domain.capitalize()),
                "",
                "- Metadata coverage: **{:.4f}%**".format(gate["metadata_coverage_percent"]),
                "- Gate G1: **{}**".format(gate["status"]),
                "- 5-core users: {} → {}".format(core["users_before"], core["users_after"]),
                "- 5-core items: {} → {}".format(core["items_before"], core["items_after"]),
                "- 5-core interactions: {} → {}".format(
                    core["interactions_before"], core["interactions_after"]
                ),
                "- Số vòng hội tụ: {}".format(core["iterations_to_converge"]),
                "- Tổng category: {}".format(taxonomy["total_categories"]),
                "- Tổng leaf sau merge: {}".format(taxonomy["total_leaf_categories"]),
                "- Leaf trước/sau merge: {} → {}".format(
                    merge["leaf_before_merge"], merge["leaf_after_merge"]
                ),
                "- Item merge lên ancestor: {}".format(merge["items_merged"]),
                "- Average taxonomy depth: {}".format(taxonomy["average_taxonomy_depth"]),
                "- Average taxonomy paths: {}".format(taxonomy["average_taxonomy_paths"]),
                "- Dataset model-ready: `{}` (seed {})".format(
                    model_ready["directory"], model_ready["seed"]
                ),
                "",
                "### Split và cold-start",
                "",
            ]
        )
        for seed, split in data["split_statistics"].items():
            cold = data["cold_start_statistics"]["seed_{}".format(seed)]
            lines.append(
                "- Seed {}: train={}, validation={}, Near-Cold={}, Long-tail={}, Warm={}".format(
                    seed,
                    split["train_interactions"],
                    split["validation_interactions"],
                    cold["near_cold"]["num_items"],
                    cold["long_tail"]["num_items"],
                    cold["warm"]["num_items"],
                )
            )
            lines.append(
                "  - Test: users={}, interactions={}, source hash match={}".format(
                    split["test_users"],
                    split["test_interactions"],
                    split["test_matches_source"],
                )
            )
        lines.extend(["", "### Gate G1", ""])
        lines.extend("- {}".format(reason) for reason in gate["reasons"])
        lines.append("")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError("Không tìm thấy file bắt buộc: {}".format(path))


def _validate_configuration(
    seeds: Sequence[int],
    dataset_seed: int,
    ratio: float,
    core_size: int,
    min_leaf_size: int,
    threshold: float,
) -> None:
    if not 0.0 < ratio < 1.0:
        raise ValueError("validation_ratio phải nằm trong (0, 1)")
    if core_size < 1 or min_leaf_size < 1:
        raise ValueError("core_size và min_leaf_size phải dương")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("coverage_threshold phải nằm trong [0, 1]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Danh sách seed không được trùng")
    if dataset_seed not in seeds:
        raise ValueError("dataset_seed phải có trong danh sách --seeds")


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    sys.exit(main())
