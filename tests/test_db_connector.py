import logging
import sqlite3
from unittest.mock import MagicMock

import pytest
import numpy as np

from services.logger.logger import setup_logging
from utils.db_connector import DbConnector


IMG_NAME = "HX-14365_073_001_14822.tif"
@pytest.fixture(scope="session",autouse=True)
def setup():
    setup_logging()


@pytest.fixture(autouse=True)
def reset_db_singleton(monkeypatch):
    DbConnector._instance = None
    DbConnector._conn = None
    monkeypatch.setattr(DbConnector, "_get_db_path", staticmethod(lambda: ":memory:"))
    yield
    if DbConnector._conn is not None:
        DbConnector._conn.close()
    DbConnector._instance = None
    DbConnector._conn = None


def test_singleton():
    a = DbConnector()
    b = DbConnector()
    assert a is b


def test_connection_established():
    db = DbConnector()
    assert db._conn is not None



def test_add_image():
    db = DbConnector()
    db.add_image(IMG_NAME)
    row = db._conn.execute(
        "SELECT img_id, prefix, line, line_number, abs_number FROM images WHERE img_id = ?",
        (IMG_NAME,)
    ).fetchone()
    assert row is not None
    assert row[0] == IMG_NAME
    assert row[1] == "HX-14365"
    assert row[2] == 73
    assert row[3] == 1
    assert row[4] == 14822


def test_add_image_duplicate_raises():
    db = DbConnector()
    db.add_image(IMG_NAME)
    result = db.add_image(IMG_NAME)
    assert result is False


def test_add_artifact_data():
    db = DbConnector()
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    db.add_artifact_data(IMG_NAME, data, offset=10)
    row = db._conn.execute(
        "SELECT img_id, dtype, shape, offset FROM artifact_datapoints WHERE img_id LIKE ?",
        (IMG_NAME,)
    ).fetchone()
    assert row is not None
    assert row[0] == IMG_NAME
    assert row[3] == 10


def test_add_artifact_data_blob_roundtrip():
    db = DbConnector()
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    db.add_artifact_data(IMG_NAME, data, offset=10)
    row = db._conn.execute(
        "SELECT dtype, shape, data FROM artifact_datapoints WHERE img_id = ?",
        (IMG_NAME,)
    ).fetchone()
    restored = np.frombuffer(row[2], dtype=row[0]).reshape(eval(row[1]))
    np.testing.assert_array_equal(restored, data)


def test_duplicate_image_logs_error(caplog):
    db = DbConnector()
    db.add_image(IMG_NAME)
    with caplog.at_level(logging.ERROR, logger="database.connector"):
        db.add_image(IMG_NAME)
    assert any("write to database" in r.message for r in caplog.records)


def test_add_artifact_data_logs_on_db_error(caplog):
    db = DbConnector()
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = sqlite3.DatabaseError("forced error")
    db._conn = mock_conn
    with caplog.at_level(logging.ERROR, logger="database.connector"):
        db.add_artifact_data(IMG_NAME, np.array([1.0], dtype=np.float32), offset=0)
    record = caplog.records[0]
    assert record.levelno == logging.ERROR
    assert "forced error" in record.message


def test_add_artifact_candidate():
    db = DbConnector()
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    db.add_artifact_data(IMG_NAME, data, offset=10)
    db.add_artifact_candidate(IMG_NAME, color=0.5, diff=0.1, offset=10, coords=(5, 10))
    row = db._conn.execute(
        "SELECT coord_x, coord_y, color_value, diff_value FROM artifact_candidates WHERE img_id = ?",
        (IMG_NAME,)
    ).fetchone()
    assert row is not None
    assert row[0] == 5
    assert row[1] == 10
    assert row[2] == pytest.approx(0.5)
    assert row[3] == pytest.approx(0.1)

