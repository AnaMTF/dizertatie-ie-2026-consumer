import numpy as np
from PIL import Image
import tensorflow as tf

import config


class UnsupportedRouteError(ValueError):
    pass


class ModelRegistry:
    def __init__(
        self,
        model_catalog,
        exact_mapping,
        image_type_mapping=None,
        default_model_key="",
        covid_route_keywords=None,
    ):
        self.model_catalog = model_catalog
        self.exact_mapping = exact_mapping
        self.image_type_mapping = image_type_mapping or {}
        self.default_model_key = default_model_key
        self.covid_route_keywords = tuple(covid_route_keywords or ())
        self.models = {}

    def _normalize_model_keys(self, mapped_value):
        if isinstance(mapped_value, str):
            keys = [mapped_value]
        elif isinstance(mapped_value, (list, tuple, set)):
            keys = [item for item in mapped_value if isinstance(item, str) and item]
        else:
            keys = []

        deduped = []
        seen = set()

        for model_key in keys:
            if model_key in seen:
                continue
            deduped.append(model_key)
            seen.add(model_key)

        return deduped

    def _mapping_key(self, image_type, body_part):
        image = image_type or ""
        body = body_part or ""
        return f"{image}|{body}"

    def _resolve_model_config(self, model_key):
        model_config = self.model_catalog.get(model_key)

        if not model_config:
            raise UnsupportedRouteError(f"Unknown model key: {model_key}")

        return model_config

    def _maybe_override_chest_models(self, image_type, body_part, file_path, model_keys):
        if "chest_pneumonia_b3" not in model_keys:
            return model_keys

        if image_type != "X-Ray" or body_part != "Chest":
            return model_keys

        if "covid_radiography" not in self.model_catalog:
            return model_keys

        lowered_path = (file_path or "").lower()
        if any(keyword in lowered_path for keyword in self.covid_route_keywords):
            # Keep backward-compatible behavior for single-model chest routing.
            if model_keys == ["chest_pneumonia_b3"]:
                return ["covid_radiography"]

            if "covid_radiography" not in model_keys:
                return [*model_keys, "covid_radiography"]

        return model_keys

    def resolve_models(self, image_type, body_part, file_path=""):
        route_key = self._mapping_key(image_type, body_part)
        mapped_value = self.exact_mapping.get(route_key)

        if mapped_value is None:
            image_route = image_type or ""
            mapped_value = self.image_type_mapping.get(image_route)
            route_key = image_route

        model_keys = self._normalize_model_keys(mapped_value)

        if not model_keys and self.default_model_key:
            model_keys = self._normalize_model_keys(self.default_model_key)
            route_key = "default"

        if not model_keys:
            raise UnsupportedRouteError(
                f"Unsupported model mapping for imageType={image_type!r}, bodyPart={body_part!r}"
            )

        model_keys = self._maybe_override_chest_models(
            image_type=image_type,
            body_part=body_part,
            file_path=file_path,
            model_keys=model_keys,
        )

        resolved = []
        for model_key in model_keys:
            model_config = self._resolve_model_config(model_key)
            resolved.append(
                {
                    "routeKey": route_key,
                    "modelKey": model_key,
                    "modelConfig": model_config,
                }
            )

        return resolved

    def resolve_model(self, image_type, body_part, file_path=""):
        resolved = self.resolve_models(
            image_type=image_type,
            body_part=body_part,
            file_path=file_path,
        )
        first = resolved[0]
        return first["routeKey"], first["modelKey"], first["modelConfig"]

    def missing_model_paths(self):
        missing = []

        for model_key, model_config in self.model_catalog.items():
            model_path = model_config.get("path", "")
            if not model_path:
                missing.append({"modelKey": model_key, "path": model_path})
                continue

            if not tf.io.gfile.exists(model_path):
                missing.append({"modelKey": model_key, "path": model_path})

        return missing

    def get(self, model_key):
        model_config = self._resolve_model_config(model_key)
        model_path = model_config.get("path", "")

        if not model_path:
            raise FileNotFoundError(f"Model path missing for modelKey={model_key!r}")

        if not tf.io.gfile.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found for modelKey={model_key!r} at path={model_path!r}"
            )

        cache_key = f"{model_key}:{model_path}"

        if cache_key not in self.models:
            self.models[cache_key] = tf.keras.models.load_model(model_path)

        return self.models[cache_key]


def _input_size_from_model(model):
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if not shape or len(shape) < 3:
        return config.DEFAULT_INPUT_HEIGHT, config.DEFAULT_INPUT_WIDTH

    height = shape[1] if isinstance(shape[1], int) and shape[1] else config.DEFAULT_INPUT_HEIGHT
    width = shape[2] if isinstance(shape[2], int) and shape[2] else config.DEFAULT_INPUT_WIDTH

    return height, width


def preprocess_image(file_path, model, preprocessing="efficientnet"):
    target_height, target_width = _input_size_from_model(model)

    image = Image.open(file_path).convert("RGB")
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    array = np.array(image, dtype=np.float32)

    if preprocessing == "efficientnet":
        array = tf.keras.applications.efficientnet.preprocess_input(array)
    elif preprocessing == "unit_scale":
        array = array / 255.0
    else:
        raise ValueError(f"Unsupported preprocessing strategy: {preprocessing!r}")

    return np.expand_dims(array, axis=0)


def _decode_prediction(logits, task, class_names):
    values = np.array(logits, dtype=np.float32).reshape(-1)

    if values.size == 0:
        raise ValueError("Model prediction returned an empty tensor")

    safe_names = list(class_names or [])

    if task == "binary" or values.size == 1:
        positive_prob = float(np.clip(values[0], 0.0, 1.0))
        negative_prob = float(1.0 - positive_prob)

        if len(safe_names) >= 2:
            negative_label, positive_label = safe_names[0], safe_names[1]
        else:
            negative_label, positive_label = "negative", "positive"

        probabilities = {
            negative_label: negative_prob,
            positive_label: positive_prob,
        }

        top_label = positive_label if positive_prob >= 0.5 else negative_label
        confidence = probabilities[top_label]

        return {
            "label": top_label,
            "confidence": float(confidence),
            "probabilities": probabilities,
        }

    top_index = int(np.argmax(values))

    if safe_names and len(safe_names) == len(values):
        top_label = safe_names[top_index]
        probabilities = {
            safe_names[index]: float(value)
            for index, value in enumerate(values)
        }
    else:
        top_label = f"class_{top_index}"
        probabilities = {
            f"class_{index}": float(value)
            for index, value in enumerate(values)
        }

    return {
        "label": top_label,
        "confidence": float(values[top_index]),
        "probabilities": probabilities,
    }


def predict_image(file_path, model, model_config):
    preprocessing = model_config.get("preprocessing", "efficientnet")
    class_names = model_config.get("class_names") or []
    task = model_config.get("task", "multiclass")

    tensor = preprocess_image(file_path, model, preprocessing=preprocessing)
    prediction = model.predict(tensor, verbose=0)
    return _decode_prediction(prediction[0], task=task, class_names=class_names)
