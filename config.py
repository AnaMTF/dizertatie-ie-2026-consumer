import os

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "app")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "app")
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_HEARTBEAT = int(os.getenv("RABBITMQ_HEARTBEAT", "60"))
RABBITMQ_BLOCKED_TIMEOUT = int(os.getenv("RABBITMQ_BLOCKED_TIMEOUT", "120"))

JOB_QUEUE = os.getenv("RABBITMQ_JOB_QUEUE", "scan.jobs.v1")
RESULT_QUEUE = os.getenv("RABBITMQ_RESULT_QUEUE", "scan.results.v1")
DLX = os.getenv("RABBITMQ_DLX", "scan.dlx")
JOB_DLQ = os.getenv("RABBITMQ_JOB_DLQ", "scan.jobs.v1.dlq")
RESULT_DLQ = os.getenv("RABBITMQ_RESULT_DLQ", "scan.results.v1.dlq")
PREFETCH_COUNT = int(os.getenv("WORKER_PREFETCH", "1"))

DEFAULT_INPUT_WIDTH = int(os.getenv("DEFAULT_INPUT_WIDTH", "224"))
DEFAULT_INPUT_HEIGHT = int(os.getenv("DEFAULT_INPUT_HEIGHT", "224"))

DEFAULT_MODEL_KEY = os.getenv("DEFAULT_MODEL_KEY", "")

# Canonical model catalog. Keep file paths relative to the consumer project root.
MODEL_CATALOG = {
    "brain_mri_b3": {
        "path": "./models/brain_mri_classifier_b3.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
    },
    "chest_pneumonia_b3": {
        "path": "./models/chest_pneumonia_classifier_b3.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": ["normal", "pneumonia"],
    },
    "covid_radiography": {
        "path": "./models/covid_radiography_classifier.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": ["covid", "lung_opacity", "normal", "viral_pneumonia"],
    },
    "kidney_ct": {
        "path": "./models/kidney_ct_classifier.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": ["cyst", "normal", "stone", "tumor"],
    },
    "breast_cancer": {
        "path": "./models/breast_cancer_best.keras",
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": ["benign", "malignant"],
    },
    "retinal_oct": {
        "path": "./models/retinal_oct_classifier_finetuned.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": [],
    },
    "oral_cancer": {
        "path": "./models/oral_cancer_classifier.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": [],
    },
    "skin": {
        "path": "./models/skin_best.keras",
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": [],
    },
}

# Strict explicit route mapping from backend payload enums.
MODEL_MAPPING_EXACT = {
    "MRI|Head / Brain": "brain_mri_b3",
    "CT Scan|Abdomen": "kidney_ct",
    "X-Ray|Chest": ["chest_pneumonia_b3", "covid_radiography"],
    "Mammography|Chest": "breast_cancer",
}

# Fallback by image type when body part granularity is too broad.
# Each mapping value can be a model key string or a list of model keys.
MODEL_MAPPING_BY_IMAGE_TYPE = {
    "Mammography": "breast_cancer",
}

# Chest route override: pneumonia is default, covid is selected when file name indicates covid-like studies.
COVID_ROUTE_KEYWORDS = tuple(
    keyword.strip().lower()
    for keyword in os.getenv(
        "COVID_ROUTE_KEYWORDS",
        "covid,sars,opacity,viral",
    ).split(",")
    if keyword.strip()
)
