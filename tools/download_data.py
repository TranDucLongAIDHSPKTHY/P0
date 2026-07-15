"""Tải dataset LightGCN và metadata Amazon/Yelp cần cho TaxPro-CL."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

from utils import configure_logging, download_file, extract_gzip, extract_zip
from config_path.config_path import (
    AMAZON_METADATA_URLS,
    DATASET_FILES,
    DATASET_REMOTE_FILES,
    LIGHTGCN_RAW_URL,
    LIGHTGCN_YELP_SOURCE_URL,
    PROJECT_ROOT,
    README_FILE_NAME,
    YELP_DATASET_NAME,
    YELP_METADATA_URLS,
    amazon_metadata_year_dir,
    dataset_directory,
    metadata_root_for,
    preprocessing_log_path,
    yelp_metadata_archive,
    yelp_metadata_root,
    yelp_metadata_year_dir,
)

YELP_README = """# Yelp2018

Dữ liệu đã xử lý dùng cho LightGCN, tải từ:
{source_url}

Các file `train.txt`, `test.txt`, `item_list.txt` và `user_list.txt` được giữ nguyên
tên và cấu trúc của nguồn LightGCN.
""".format(source_url=LIGHTGCN_YELP_SOURCE_URL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tải dataset LightGCN và metadata Amazon/Yelp cho ID-GRec."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại và giải nén lại kể cả khi đầu ra đã tồn tại.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Thư mục gốc dự án (mặc định: thư mục cha của tools).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Mức log console (mặc định: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="File log tiền xử lý (mặc định: <root>/preprocessing_logs/preprocessing.log).",
    )
    return parser.parse_args()


def download_datasets(root: Path, force: bool, logger: logging.Logger) -> None:
    logger.info("Bắt đầu tải dataset LightGCN")
    for dataset, remote_files in DATASET_REMOTE_FILES.items():
        dataset_dir = dataset_directory(root, dataset)
        for filename in remote_files:
            download_file(
                "{}/{}/{}".format(LIGHTGCN_RAW_URL, dataset, filename),
                dataset_dir / filename,
                force=force,
                logger=logger,
            )
        if dataset == YELP_DATASET_NAME:
            _write_yelp_readme(dataset_dir / README_FILE_NAME, force, logger)
    _require_files(root, _dataset_outputs(root), "dataset LightGCN")
    logger.info("Dataset LightGCN đã đầy đủ")


def download_metadata(root: Path, force: bool, logger: logging.Logger) -> None:
    logger.info("Bắt đầu tải metadata Amazon")
    for year, url in AMAZON_METADATA_URLS.items():
        year_dir = amazon_metadata_year_dir(root, year)
        archive = year_dir / Path(url).name
        output = year_dir / archive.name[:-3]
        download_file(url, archive, force=force, logger=logger)
        extract_gzip(archive, output, force=force, logger=logger)

    logger.info("Bắt đầu tải metadata Yelp")
    yelp_root = yelp_metadata_root(root)
    for year, url in YELP_METADATA_URLS.items():
        archive = yelp_metadata_archive(root, year)
        output_dir = yelp_metadata_year_dir(root, year)
        download_file(url, archive, force=force, logger=logger)
        extract_zip(archive, output_dir, force=force, logger=logger)

    _require_files(root, _amazon_outputs(root), "metadata Amazon")
    for year in YELP_METADATA_URLS:
        output_dir = yelp_metadata_year_dir(root, year)
        if not output_dir.is_dir() or not any(path.is_file() for path in output_dir.rglob("*")):
            raise RuntimeError("Metadata Yelp {} chưa được giải nén: {}".format(year, output_dir))
    logger.info("Metadata Amazon và Yelp đã tải, giải nén đầy đủ")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    log_file = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else preprocessing_log_path(root, "pipeline")
    )
    logger = configure_logging(args.log_level, log_file, "download_data")
    logger.info("========== BẮT ĐẦU DOWNLOAD DATA ==========")
    logger.info("File log tiền xử lý: %s", log_file)
    logger.info("Thư mục dự án: %s", root)
    logger.info("Chế độ --force: %s", "bật" if args.force else "tắt")
    try:
        download_datasets(root, args.force, logger)
        download_metadata(root, args.force, logger)
    except Exception as error:
        logger.exception("Chuẩn bị dữ liệu thất bại: %s", error)
        return 1
    logger.info("Hoàn tất toàn bộ bước chuẩn bị dữ liệu")
    logger.info("========== KẾT THÚC DOWNLOAD DATA ==========")
    return 0


def _write_yelp_readme(path: Path, force: bool, logger: logging.Logger) -> None:
    if path.is_file() and not force:
        logger.info("Bỏ qua file đã tồn tại: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(YELP_README, encoding="utf-8")
    logger.info("Đã tạo README cho Yelp2018: %s", path)


def _dataset_outputs(root: Path) -> Iterable[Path]:
    for dataset in DATASET_REMOTE_FILES:
        for filename in DATASET_FILES:
            yield dataset_directory(root, dataset) / filename


def _amazon_outputs(root: Path) -> Iterable[Path]:
    for year, url in AMAZON_METADATA_URLS.items():
        yield amazon_metadata_year_dir(root, year) / Path(url).name[:-3]


def _require_files(root: Path, relative_paths: Iterable[Path], label: str) -> None:
    missing = []
    for path in relative_paths:
        resolved = root / path
        if not resolved.is_file():
            try:
                display_path = resolved.relative_to(root)
            except ValueError:
                display_path = resolved
            missing.append(str(display_path))
    if missing:
        raise RuntimeError("Thiếu file {}: {}".format(label, ", ".join(missing)))


if __name__ == "__main__":
    sys.exit(main())
