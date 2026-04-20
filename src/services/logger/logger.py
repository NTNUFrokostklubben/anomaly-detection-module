import logging
import logging.handlers
from pathlib import Path
from queue import Queue


def _make_file_handler(filename: str, log_dir: Path, fmt: logging.Formatter) -> logging.handlers.RotatingFileHandler:
    fh = logging.handlers.RotatingFileHandler(log_dir / filename, maxBytes=10_000_000, backupCount=3)
    fh.setFormatter(fmt)
    return fh


def setup_logging(log_dir: Path = Path(__file__).parent.parent.parent.parent / 'logs', level: int = logging.INFO):
    """
    Setup logging configuration. Handles logs based on concerns, to add more concerns,
    add a new entry to the concerns dict with the name of the concern,
    the filename and the formatter to use for that concern.
    The log records will be filtered based on the logger name,
    so make sure to use logger names that start with the concern name for the logs to be handled correctly.
    :param log_dir: The directory for logs, default is project_root/logs
    :param level: the level at which the root logger should log, default is logging.INFO, higher level means lower level
    logs will be ignored.
    :return:  A listener for the queue, must be gracefully handled to ensure queued logs are not lost.
    """
    log_dir.mkdir(exist_ok=True)
    concerns = {
        "analysis": (
            "analysis.log",
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s analysis=%(analysis)s img_id=%(img_id)s | %(message)s")
        ),
        "database": (
            "database.log",
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s | op=%(operation)s table=%(table)s func=%(funcName)s | %(message)s")
        ),
        "grpc": (
            "grpc.log",
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s func=%(funcName)s | %(message)s")
        ),
    }

    real_handlers = []

    for name, (filename, fmt) in concerns.items():
        fh = _make_file_handler(filename, log_dir, fmt)
        fh.addFilter(lambda record, n=name: record.name.startswith(n))
        real_handlers.append(fh)

    # Single queue and listener — all records flow through one queue
    log_queue = Queue()
    listener = logging.handlers.QueueListener(log_queue, *real_handlers, respect_handler_level=True)
    listener.start()

    # Root logger gets a single QueueHandler
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(queue_handler)

    return listener  # caller must hold a reference and call .stop() on shutdown
