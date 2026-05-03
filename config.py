import os


CONSUMER_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CONSUMER_ROOT, "models")


def _model_path(file_name):
    return os.path.join(MODELS_DIR, file_name)


def _load_class_names(file_name, fallback=None):
    if not file_name:
        return list(fallback or [])

    path = _model_path(file_name)
    if not os.path.exists(path):
        return list(fallback or [])

    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


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
DEFAULT_MODEL_KEY = ""

MODEL_CATALOG = {
    "brain_mri_b3": {
        "path": _model_path("brain_mri_classifier_b3.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names(
            "brain_mri_class_names.txt",
            fallback=["glioma", "meningioma", "notumor", "pituitary"],
        ),
    },
    "tb_chest_xray": {
        "path": _model_path("tb_chest_xray_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": _load_class_names(
            "tb_chest_xray_class_names.txt",
            fallback=["normal", "tuberculosis"],
        ),
    },
    "covid_radiography": {
        "path": _model_path("covid_radiography_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names(
            "covid_radiography_class_names.txt",
            fallback=["covid", "lung_opacity", "normal", "viral_pneumonia"],
        ),
    },
    "chest_pneumonia_b3": {
        "path": _model_path("chest_pneumonia_classifier_b3.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names(
            "chest_pneumonia_class_names.txt",
            fallback=["normal", "pneumonia"],
        ),
    },
    "kidney_ct": {
        "path": _model_path("kidney_ct_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names(
            "kidney_ct_class_names.txt",
            fallback=["cyst", "normal", "stone", "tumor"],
        ),
    },
    "liver_fatty_ct": {
        "path": _model_path("liver_fatty_ct_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": _load_class_names("liver_fatty_ct_class_names.txt"),
    },
    "liver_malignant_binary": {
        "path": _model_path("liver_malignant_binary_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": _load_class_names("liver_malignant_binary_class_names.txt"),
    },
    "ovarian_cyst": {
        "path": _model_path("ovarian_cyst_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("ovarian_cyst_class_names.txt"),
    },
    "knee_xray_osteoarthritis": {
        "path": _model_path("knee_xray_osteoarthritis_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("knee_xray_osteoarthritis_class_names.txt"),
    },
    "cervical_cancer": {
        "path": _model_path("cervical_cancer_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("cervical_cancer_class_names.txt"),
    },
    "lung_colon_cancer": {
        "path": _model_path("lung_colon_cancer_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("lung_colon_cancer_class_names.txt"),
    },
    "lymphoma": {
        "path": _model_path("lymphoma_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("lymphoma_class_names.txt"),
    },
    "retinal_oct_finetuned": {
        "path": _model_path("retinal_oct_classifier_finetuned.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("retinal_oct_class_names.txt"),
    },
    "breast_cancer_best": {
        "path": _model_path("breast_cancer_best.keras"),
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": _load_class_names(
            None,
            fallback=["benign", "malignant"],
        ),
    },
    "breast_ultrasound": {
        "path": _model_path("breast_ultrasound_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("breast_ultrasound_class_names.txt"),
    },
    "oral_cancer_b3": {
        "path": _model_path("oral_cancer_classifier_b3.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("oral_cancer_class_names.txt"),
    },
    "colonoscopy_finetuned": {
        "path": _model_path("colonoscopy_classifier_finetuned.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("colonoscopy_class_names.txt"),
    },
    "skin_best": {
        "path": _model_path("skin_best.keras"),
        "preprocessing": "efficientnet",
        "task": "multiclass",
        "class_names": _load_class_names("class_names.txt"),
    },
    "heart_abnormal_binary": {
        "path": _model_path("heart_abnormal_binary_classifier.keras"),
        "preprocessing": "efficientnet",
        "task": "binary",
        "class_names": _load_class_names("heart_abnormal_binary_class_names.txt"),
    },
}

MODEL_MAPPING_EXACT = {
    "MRI|Brain": "brain_mri_b3",
    "X-Ray|Chest / lungs": [
        "tb_chest_xray",
        "covid_radiography",
    ],
    "CT Scan|Kidney": "kidney_ct",
    "CT Scan|Liver": "liver_fatty_ct",
    "Ultrasound|Liver": "liver_malignant_binary",
    "Ultrasound|Ovary": "ovarian_cyst",
    "X-Ray|Knee": "knee_xray_osteoarthritis",
    "Colposcopy|Cervix": "cervical_cancer",
    "Histopathology|Chest / lungs": "lung_colon_cancer",
    "Histopathology|Colon": "lung_colon_cancer",
    "Histopathology|Lymph node": "lymphoma",
    "OCT|Retina": "retinal_oct_finetuned",
    "Histopathology|Breast": "breast_cancer_best",
    "Ultrasound|Breast": "breast_ultrasound",
    "Clinical photo|Oral cavity": "oral_cancer_b3",
    "Dermoscopy|Skin": "skin_best",
    "Echocardiography|Heart": "heart_abnormal_binary",
    "Endoscopy|Colon": "colonoscopy_finetuned",
}

MODEL_MAPPING_BY_IMAGE_TYPE = {}
COVID_ROUTE_KEYWORDS = ()
