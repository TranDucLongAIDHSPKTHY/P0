"""Cấu hình tập trung toàn bộ đường dẫn và nguồn dữ liệu của dự án.

Chỉ chỉnh sửa các khai báo trong file này khi cần di chuyển dữ liệu, log,
checkpoint hoặc output sang vị trí khác.
"""

from pathlib import Path


# Gốc dự án và các thư mục cấp cao.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURE_DIR = PROJECT_ROOT / "configure"
DATASET_DIR = PROJECT_ROOT / "dataset"

# Khi chạy trên server set để dữ liệu -> HDD
# /data/phuongtran/dataset_verify
# /data/phuongtran/metadata
# /data/phuongtran/preprocessed 
# Khi không chạy trên server set : bỏ /data/phuongtran/
DATASET_VERIFY_DIR = PROJECT_ROOT / "dataset_verify"
METADATA_DIR = PROJECT_ROOT / "metadata"
TAXONOMY_DIR = METADATA_DIR
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"


PREPROCESSING_LOG_DIR = PROJECT_ROOT / "preprocessing_logs"
TRAINING_LOG_DIR = PROJECT_ROOT / "training_logs"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "log"
RESULT_DIR = MODEL_OUTPUT_DIR
CHECKPOINT_DIR = MODEL_OUTPUT_DIR
CACHE_DIR = DATASET_VERIFY_DIR
EMBEDDING_DIR = MODEL_OUTPUT_DIR
PROTOTYPE_DIR = MODEL_OUTPUT_DIR

# Tên dataset và tên file dữ liệu chuẩn.
AMAZON_DATASET_NAME = "amazon-book"
YELP_DATASET_NAME = "yelp2018"
README_FILE_NAME = "README.md"
TRAIN_FILE_NAME = "train.txt"
VALIDATION_FILE_NAME = "validation.txt"
TEST_FILE_NAME = "test.txt"
ITEM_LIST_FILE_NAME = "item_list.txt"
USER_LIST_FILE_NAME = "user_list.txt"
DATASET_FILES = (
    README_FILE_NAME,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    ITEM_LIST_FILE_NAME,
    USER_LIST_FILE_NAME,
)
DATASET_REMOTE_FILES = {
    AMAZON_DATASET_NAME: DATASET_FILES,
    YELP_DATASET_NAME: (
        TRAIN_FILE_NAME,
        TEST_FILE_NAME,
        ITEM_LIST_FILE_NAME,
        USER_LIST_FILE_NAME,
    ),
}

# URL nguồn tải dữ liệu/metadata.
LIGHTGCN_RAW_URL = "https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data"
LIGHTGCN_YELP_SOURCE_URL = (
    "https://github.com/gusye1234/LightGCN-PyTorch/tree/master/data/yelp2018"
)
AMAZON_METADATA_URLS = {
    "2014": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Books.json.gz",
    "2018": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Books.json.gz",
    "2023": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz",
}
YELP_METADATA_URLS = {
    "2018": "https://zenodo.org/records/10998102/files/Yelp2018.zip?download=1",
    "2021": "https://zenodo.org/records/10998102/files/Yelp2021.zip?download=1",
    "2022": "https://zenodo.org/records/10998102/files/Yelp2022.zip?download=1",
}

# Metadata source và output tương đối theo project root.
AMAZON_METADATA_SOURCES = {
    "2014": METADATA_DIR / "amazon" / "2014" / "meta_Books.json",
    "2018": METADATA_DIR / "amazon" / "2018" / "meta_Books.json",
    "2023": METADATA_DIR / "amazon" / "2023" / "meta_Books.jsonl",
}
YELP_METADATA_SOURCES = {
    "2018": METADATA_DIR / "yelp" / "2018" / "Yelp2018.item",
    "2021": METADATA_DIR / "yelp" / "2021" / "Yelp2021.item",
    "2022": METADATA_DIR / "yelp" / "2022" / "Yelp2022.item",
}
MERGE_OUTPUT_NAMES = {
    "amazon": {
        "metadata": "merged_metadata_amazon.json",
        "categories": "item2category_amazon.json",
        "decisions": "merge_decisions_amazon.json",
        "summary": "merge_summary_amazon.json",
    },
    "yelp": {
        "metadata": "merged_metadata_yelp.json",
        "categories": "item2category_yelp.json",
        "decisions": "merge_decisions_yelp.json",
        "summary": "merge_summary_yelp.json",
    },
}
PREPROCESSING_DOMAIN_SOURCES = {
    "amazon": {
        "train": DATASET_DIR / AMAZON_DATASET_NAME / TRAIN_FILE_NAME,
        "test": DATASET_DIR / AMAZON_DATASET_NAME / TEST_FILE_NAME,
        "merged": METADATA_DIR / MERGE_OUTPUT_NAMES["amazon"]["metadata"],
        "item2category": METADATA_DIR / MERGE_OUTPUT_NAMES["amazon"]["categories"],
        "verify_name": AMAZON_DATASET_NAME,
    },
    "yelp": {
        "train": DATASET_DIR / YELP_DATASET_NAME / TRAIN_FILE_NAME,
        "test": DATASET_DIR / YELP_DATASET_NAME / TEST_FILE_NAME,
        "merged": METADATA_DIR / MERGE_OUTPUT_NAMES["yelp"]["metadata"],
        "item2category": METADATA_DIR / MERGE_OUTPUT_NAMES["yelp"]["categories"],
        "verify_name": YELP_DATASET_NAME,
    },
}

# Tên file log, cache, checkpoint và result.
PREPROCESSING_LOG_FILE_NAME = "preprocessing.log"
PREPARE_METADATA_LOG_FILE_NAME = "prepare_metadata.log"
DOWNLOAD_DATA_LOG_FILE_NAME = "download_data.log"
TRAINING_LOG_FILE_NAME = "training.log"
PREPROCESSING_SUMMARY_JSON_NAME = "preprocessing_summary.json"
PREPROCESSING_SUMMARY_MD_NAME = "preprocessing_summary.md"
FIVE_CORE_TRAIN_FILE_NAME = "five_core_train.txt"
VALIDATION_METRICS_FILE_NAME = "validation_metrics.json"
FINAL_TEST_METRICS_FILE_NAME = "final_test_metrics.json"
LAST_MODEL_FILE_NAME = "last_model.pt"
BEST_VALIDATION_MODEL_FILE_NAME = "best_validation_model.pt"
ADJACENCY_WITH_SELF_FILE_NAME = "pre_A_with_self.npz"
ADJACENCY_FILE_NAME = "pre_A.npz"
INTERACTION_ADJACENCY_FILE_NAME = "pre_R.npz"
ATOMIC_TEMPORARY_SUFFIX = ".tmp"
DOWNLOAD_TEMPORARY_SUFFIX = ".part"


def model_config_file(model_name):
    return CONFIGURE_DIR / (str(model_name) + ".txt")


def verified_dataset_dir(dataset_name):
    return DATASET_VERIFY_DIR / str(dataset_name)


def dataset_split_file(dataset_directory, split):
    names = {
        "train": TRAIN_FILE_NAME,
        "validation": VALIDATION_FILE_NAME,
        "test": TEST_FILE_NAME,
    }
    return Path(dataset_directory) / names[split]


def model_result_dir(model_name, dataset_name):
    return MODEL_OUTPUT_DIR / str(model_name) / str(dataset_name)


def training_log_file(model_name):
    return TRAINING_LOG_DIR / str(model_name) / TRAINING_LOG_FILE_NAME


def checkpoint_file(output_dir, checkpoint_kind):
    names = {
        "last": LAST_MODEL_FILE_NAME,
        "best_validation": BEST_VALIDATION_MODEL_FILE_NAME,
    }
    return Path(output_dir) / names[checkpoint_kind]


def result_file(output_dir, result_kind):
    names = {
        "validation": VALIDATION_METRICS_FILE_NAME,
        "final_test": FINAL_TEST_METRICS_FILE_NAME,
    }
    return Path(output_dir) / names[result_kind]


def adjacency_cache_file(dataset_directory, cache_kind, alpha=None, beta=None, suffix=True):
    names = {
        "with_self": ADJACENCY_WITH_SELF_FILE_NAME,
        "adjacency": ADJACENCY_FILE_NAME,
        "interaction": INTERACTION_ADJACENCY_FILE_NAME,
    }
    if cache_kind == "lightgcn_pp":
        name = "pre_A_{}_{}.npz".format(alpha, beta)
    else:
        name = names[cache_kind]
    dataset_directory = Path(dataset_directory)
    try:
        relative_dataset = dataset_directory.resolve().relative_to(DATASET_VERIFY_DIR.resolve())
        cache_directory = CACHE_DIR / relative_dataset
    except ValueError:
        cache_directory = dataset_directory
    path = cache_directory / name
    return path if suffix else path.with_suffix("")


def configured_for_root(root, configured_path, default_relative):
    """Giữ override --root cũ, đồng thời dùng path cấu hình khi root là mặc định."""
    root = Path(root).expanduser().resolve()
    if root == PROJECT_ROOT.expanduser().resolve():
        return Path(configured_path)
    return root / Path(default_relative)


def dataset_root_for(root):
    return configured_for_root(root, DATASET_DIR, "dataset")


def metadata_root_for(root):
    return configured_for_root(root, METADATA_DIR, "metadata")


def preprocessed_root_for(root):
    return configured_for_root(root, PREPROCESSED_DIR, "preprocessed")


def dataset_verify_root_for(root):
    return configured_for_root(root, DATASET_VERIFY_DIR, "dataset_verify")


def preprocessing_log_root_for(root):
    return configured_for_root(root, PREPROCESSING_LOG_DIR, "preprocessing_logs")


def dataset_directory(root, dataset_name):
    return dataset_root_for(root) / str(dataset_name)


def amazon_metadata_year_dir(root, year):
    return metadata_root_for(root) / "amazon" / str(year)


def yelp_metadata_root(root):
    return metadata_root_for(root) / "yelp"


def yelp_metadata_year_dir(root, year):
    return yelp_metadata_root(root) / str(year)


def yelp_metadata_archive(root, year):
    return yelp_metadata_root(root) / "Yelp{}.zip".format(year)


def preprocessing_log_path(root, log_kind):
    names = {
        "pipeline": PREPROCESSING_LOG_FILE_NAME,
        "prepare_metadata": PREPARE_METADATA_LOG_FILE_NAME,
        "download": DOWNLOAD_DATA_LOG_FILE_NAME,
    }
    return preprocessing_log_root_for(root) / names[log_kind]


def item_mapping_file(root, domain):
    dataset_name = AMAZON_DATASET_NAME if domain == "amazon" else YELP_DATASET_NAME
    return dataset_root_for(root) / dataset_name / ITEM_LIST_FILE_NAME


def amazon_metadata_source_file(root, year):
    default_relative = Path("metadata") / "amazon" / str(year) / AMAZON_METADATA_SOURCES[str(year)].name
    return configured_for_root(root, AMAZON_METADATA_SOURCES[str(year)], default_relative)


def yelp_metadata_source_file(root, year):
    default_relative = Path("metadata") / "yelp" / str(year) / YELP_METADATA_SOURCES[str(year)].name
    return configured_for_root(root, YELP_METADATA_SOURCES[str(year)], default_relative)


def preprocessing_domain_sources_for(root):
    sources = {}
    for domain, values in PREPROCESSING_DOMAIN_SOURCES.items():
        dataset_name = values["verify_name"]
        sources[domain] = {
            "train": dataset_directory(root, dataset_name) / TRAIN_FILE_NAME,
            "test": dataset_directory(root, dataset_name) / TEST_FILE_NAME,
            "merged": metadata_root_for(root) / MERGE_OUTPUT_NAMES[domain]["metadata"],
            "item2category": metadata_root_for(root) / MERGE_OUTPUT_NAMES[domain]["categories"],
            "verify_name": dataset_name,
        }
    return sources


def taxonomy_distribution_file(output_dir, domain):
    return Path(output_dir) / "taxonomy_distribution_{}.png".format(domain)


def split_seed_dir(domain_dir, seed):
    return Path(domain_dir) / "splits" / "seed_{}".format(seed)


def preprocessing_summary_file(output_dir, summary_format):
    names = {
        "json": PREPROCESSING_SUMMARY_JSON_NAME,
        "markdown": PREPROCESSING_SUMMARY_MD_NAME,
    }
    return Path(output_dir) / names[summary_format]


def five_core_train_file(domain_dir):
    return Path(domain_dir) / FIVE_CORE_TRAIN_FILE_NAME


def split_output_file(seed_dir, split):
    return dataset_split_file(seed_dir, split)


def temporary_file(target, temporary_kind="atomic"):
    suffixes = {
        "atomic": ATOMIC_TEMPORARY_SUFFIX,
        "download": DOWNLOAD_TEMPORARY_SUFFIX,
    }
    target = Path(target)
    return target.with_name(target.name + suffixes[temporary_kind])
