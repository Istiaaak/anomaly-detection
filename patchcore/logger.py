# logger.py
import logging
from logging.handlers import RotatingFileHandler
import sys
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Construit un JSON minimaliste
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level":     record.levelname,
            "message":   record.getMessage(),
        }
        # Si on a des données struct, on les ajoute
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)

def get_logger(name: str = __name__, logfile: str = "logs/patchcore.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # File handler avec rotation
    fh = RotatingFileHandler(logfile, maxBytes=10_000_00, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    # Optionnel : console en clair
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
    return logger
