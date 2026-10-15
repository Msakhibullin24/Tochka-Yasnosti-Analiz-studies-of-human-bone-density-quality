import sqlite3
import threading

import pytest

from dxaqc import api
from dxaqc.store import Store


def test_worker_recovers_after_sqlite_busy_and_finishes_next_job(tmp_path, monkeypatch):
    repository = Store(tmp_path)
    repository.add('after-lock', tmp_path/'input', tmp_path/'out')
    claim = repository.claim
    attempts = []
    stop = threading.Event()

    def temporarily_locked():
        attempts.append(1)
        if len(attempts) == 1:
            error = sqlite3.OperationalError('database is locked')
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return claim()

    def batch(*args, **kwargs):
        stop.set()
        return {'files': 1, 'success': 1, 'failure': 0}

    monkeypatch.setattr(repository, 'claim', temporarily_locked)
    monkeypatch.setattr('dxaqc.source_archive.retain_input', lambda path, parent: path)
    monkeypatch.setattr(api, 'analyzer', lambda: object())
    monkeypatch.setattr(api, 'run_batch', batch)
    worker = threading.Thread(target=api._worker, args=(repository, stop))
    worker.start()
    worker.join(5)
    stop.set()
    assert not worker.is_alive()
    assert len(attempts) == 2
    assert repository.get('after-lock')['status'] == 'finished'
    assert repository.get('after-lock')['summary']['success'] == 1


def test_worker_does_not_hide_an_unrelated_sqlite_failure():
    class BrokenStore:
        def claim(self):
            error = sqlite3.OperationalError('no such table: jobs')
            error.sqlite_errorcode = sqlite3.SQLITE_ERROR
            raise error
    with pytest.raises(sqlite3.OperationalError, match='no such table'):
        api._worker(BrokenStore(), threading.Event())


def test_idle_claim_remains_read_only_while_another_connection_writes(tmp_path):
    repository = Store(tmp_path)
    finished = threading.Event()
    results = []

    def claim():
        results.append(repository.claim())
        finished.set()

    with repository.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        thread = threading.Thread(target=claim)
        thread.start()
        returned_during_write = finished.wait(1)
    thread.join(5)
    assert returned_during_write
    assert results == [None]


def test_status_polling_reuses_the_lifespan_repository(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    constructed = []
    original = api.Store

    def create(root):
        constructed.append(root)
        return original(root)

    monkeypatch.setattr(api, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(api, 'Store', create)
    with TestClient(api.app) as client:
        for _ in range(10):
            assert client.get('/api/v1/jobs').status_code == 200
    assert constructed == [tmp_path]
