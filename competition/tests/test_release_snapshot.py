import hashlib

from release_snapshot import comparable, sha256


def test_snapshot_verification_ignores_only_capture_metadata(tmp_path):
    recorded = {'files': {'model': {'sha256': 'old'}}, 'git_head': 'a',
                'working_tree_dirty_at_capture': True}
    current = {'files': {'model': {'sha256': 'old'}}, 'git_head': 'b',
               'working_tree_dirty_at_capture': False}
    assert comparable(recorded) == comparable(current)
    current['files']['model']['sha256'] = 'new'
    assert comparable(recorded) != comparable(current)


def test_artifact_hash_is_streamed(tmp_path):
    artifact = tmp_path / 'large.bin'
    artifact.write_bytes(b'abc' * 100_000)
    assert sha256(artifact) == hashlib.sha256(b'abc' * 100_000).hexdigest()
