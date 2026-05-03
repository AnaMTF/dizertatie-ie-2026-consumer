import json
import logging
import os
import time
from datetime import datetime, timezone

import pika
from pika.credentials import PlainCredentials
from pika.exceptions import AMQPConnectionError

import config
from inference import ModelRegistry, predict_image

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("scan-worker")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def setup_infrastructure(channel):
    channel.exchange_declare(exchange=config.DLX, exchange_type="direct", durable=True)

    channel.queue_declare(queue=config.JOB_DLQ, durable=True)
    channel.queue_bind(queue=config.JOB_DLQ, exchange=config.DLX, routing_key=config.JOB_DLQ)

    channel.queue_declare(queue=config.RESULT_DLQ, durable=True)
    channel.queue_bind(queue=config.RESULT_DLQ, exchange=config.DLX, routing_key=config.RESULT_DLQ)

    channel.queue_declare(
        queue=config.JOB_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": config.DLX,
            "x-dead-letter-routing-key": config.JOB_DLQ,
        },
    )

    channel.queue_declare(
        queue=config.RESULT_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": config.DLX,
            "x-dead-letter-routing-key": config.RESULT_DLQ,
        },
    )


def connect_with_retry():
    delay_seconds = 1

    while True:
        try:
            credentials = PlainCredentials(config.RABBITMQ_USER, config.RABBITMQ_PASS)
            params = pika.ConnectionParameters(
                host=config.RABBITMQ_HOST,
                port=config.RABBITMQ_PORT,
                virtual_host=config.RABBITMQ_VHOST,
                credentials=credentials,
                heartbeat=config.RABBITMQ_HEARTBEAT,
                blocked_connection_timeout=config.RABBITMQ_BLOCKED_TIMEOUT,
            )
            connection = pika.BlockingConnection(params)
            logger.info("Connected to RabbitMQ at %s:%s", config.RABBITMQ_HOST, config.RABBITMQ_PORT)
            return connection
        except AMQPConnectionError as exc:
            logger.warning("RabbitMQ connection failed (%s). Retrying in %ss", exc, delay_seconds)
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 30)


def resolve_file_path(image, storage):
    file_path = image.get("filePath", "")

    if not file_path:
        raise FileNotFoundError("Image payload does not include filePath")

    if os.path.isabs(file_path):
        resolved = file_path
    else:
        uploads_base = storage.get("uploadsBasePath") if isinstance(storage, dict) else ""

        if uploads_base:
            normalized = file_path.replace("\\", "/")

            if normalized.startswith("./"):
                normalized = normalized[2:]

            if normalized.startswith("uploads/"):
                normalized = normalized[len("uploads/") :]

            resolved = os.path.join(uploads_base, normalized)
        else:
            resolved = os.path.abspath(file_path)

    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Image file not found: {resolved}")

    return resolved


def build_image_result(image, image_type, body_part, model_key, route_key, prediction):
    return {
        "imageUuid": image.get("imageUuid"),
        "imageType": image_type,
        "bodyPart": body_part,
        "modelKey": model_key,
        "routeKey": route_key,
        "prediction": prediction,
    }


def build_image_error(image, image_type, body_part, model_key, route_key, error):
    return {
        "imageUuid": image.get("imageUuid"),
        "imageType": image_type,
        "bodyPart": body_part,
        "modelKey": model_key,
        "routeKey": route_key,
        "error": str(error),
    }


def process_scan_job(payload, model_registry):
    images = payload.get("images")
    image_type = payload.get("imageType")
    body_part = payload.get("bodyPart")

    if not isinstance(images, list) or not images:
        raise ValueError("Scan job payload has no images")

    if not image_type or not body_part:
        raise ValueError("Scan job payload must include imageType and bodyPart")

    image_results = []
    image_errors = []

    for image in images:
        try:
            resolved_path = resolve_file_path(image, payload.get("storage", {}))
            resolved_models = model_registry.resolve_models(
                image_type=image_type,
                body_part=body_part,
                file_path=resolved_path,
            )

            for resolved_model in resolved_models:
                route_key = resolved_model.get("routeKey")
                model_key = resolved_model.get("modelKey")
                model_config = resolved_model.get("modelConfig")

                try:
                    model = model_registry.get(model_key)
                    prediction = predict_image(resolved_path, model, model_config)

                    image_results.append(
                        build_image_result(
                            image=image,
                            image_type=image_type,
                            body_part=body_part,
                            model_key=model_key,
                            route_key=route_key,
                            prediction=prediction,
                        )
                    )
                except Exception as exc:
                    image_errors.append(
                        build_image_error(
                            image=image,
                            image_type=image_type,
                            body_part=body_part,
                            model_key=model_key,
                            route_key=route_key,
                            error=exc,
                        )
                    )
        except Exception as exc:
            image_errors.append(
                build_image_error(
                    image=image,
                    image_type=image_type,
                    body_part=body_part,
                    model_key=None,
                    route_key=None,
                    error=exc,
                )
            )

    total_predictions = len(image_results) + len(image_errors)

    return {
        "images": image_results,
        "count": len(image_results),
        "errors": image_errors,
        "failedCount": len(image_errors),
        "total": total_predictions,
        "imageCount": len(images),
    }


def publish_result(channel, payload):
    channel.basic_publish(
        exchange="",
        routing_key=config.RESULT_QUEUE,
        body=json.dumps(payload).encode("utf-8"),
        properties=pika.BasicProperties(
            delivery_mode=2,
            content_type="application/json",
            type="scan.result.v1",
            message_id=payload.get("jobId"),
            timestamp=int(time.time()),
        ),
    )


def build_success_payload(job_payload, results):
    return {
        "schemaVersion": "1.0",
        "eventType": "scan.result.v1",
        "jobId": job_payload.get("jobId"),
        "scanUuid": job_payload.get("scanUuid"),
        "patientUuid": job_payload.get("patientUuid"),
        "status": "completed",
        "processedAt": utc_now_iso(),
        "results": results,
    }


def build_failure_payload(job_payload, error, results=None):
    failure_results = {
        "error": "Scan classification failed",
        "details": str(error),
    }

    if isinstance(results, dict):
        failure_results = {
            **results,
            **failure_results,
        }

    return {
        "schemaVersion": "1.0",
        "eventType": "scan.result.v1",
        "jobId": job_payload.get("jobId"),
        "scanUuid": job_payload.get("scanUuid"),
        "patientUuid": job_payload.get("patientUuid"),
        "status": "failed",
        "processedAt": utc_now_iso(),
        "error": str(error),
        "results": failure_results,
    }


def run_worker():
    model_registry = ModelRegistry(
        model_catalog=config.MODEL_CATALOG,
        exact_mapping=config.MODEL_MAPPING_EXACT,
        image_type_mapping=config.MODEL_MAPPING_BY_IMAGE_TYPE,
        default_model_key=config.DEFAULT_MODEL_KEY,
        covid_route_keywords=config.COVID_ROUTE_KEYWORDS,
    )

    missing_models = model_registry.missing_model_paths()
    if missing_models:
        logger.warning("Missing model files detected: %s", missing_models)

    connection = connect_with_retry()
    channel = connection.channel()

    setup_infrastructure(channel)
    channel.basic_qos(prefetch_count=max(1, config.PREFETCH_COUNT))

    def on_message(ch, method, _properties, body):
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            logger.exception("Invalid job payload JSON")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        scan_uuid = payload.get("scanUuid")
        logger.info("Processing scan job %s", scan_uuid)

        try:
            results = process_scan_job(payload, model_registry)
            if results.get("count", 0) > 0:
                result_payload = build_success_payload(payload, results)
            else:
                result_payload = build_failure_payload(
                    payload,
                    ValueError("No images could be processed"),
                    results=results,
                )

            publish_result(ch, result_payload)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Completed scan job %s", scan_uuid)
        except Exception as exc:
            logger.exception("Failed scan job %s", scan_uuid)
            result_payload = build_failure_payload(payload, exc)
            try:
                publish_result(ch, result_payload)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception:
                logger.exception("Failed to publish failure result; requeueing")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    channel.basic_consume(queue=config.JOB_QUEUE, on_message_callback=on_message, auto_ack=False)

    logger.info("Worker consuming queue %s", config.JOB_QUEUE)

    try:
        channel.start_consuming()
    finally:
        if not connection.is_closed:
            connection.close()
