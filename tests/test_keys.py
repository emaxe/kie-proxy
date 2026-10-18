from __future__ import annotations

import pytest

from kie_proxy.config import KeyEntry
from kie_proxy.keys import KeyPool, KeyStatus, NoKeyAvailable


def make_pool(n=3) -> KeyPool:
    return KeyPool([KeyEntry(key=f"key{i}", label=f"l{i}") for i in range(1, n + 1)])


def test_round_robin():
    pool = make_pool()
    assert [pool.acquire().key for _ in range(4)] == ["key1", "key2", "key3", "key1"]


def test_rate_limited_key_is_skipped_then_recovers():
    pool = make_pool(2)
    pool.mark_rate_limited("key1", "429")
    assert pool.acquire().key == "key2"
    assert pool.acquire().key == "key2"
    # force cooldown expiry
    pool.get("key1").until = 0.0
    assert pool.acquire().key in {"key1", "key2"}


def test_invalid_key_never_used():
    pool = make_pool(2)
    pool.mark_invalid("key1")
    for _ in range(5):
        assert pool.acquire().key == "key2"


def test_exhausted_key_status():
    pool = make_pool(2)
    pool.mark_exhausted("key1")
    state = pool.get("key1")
    assert state.status == KeyStatus.EXHAUSTED
    assert pool.acquire().key == "key2"


def test_all_keys_unusable():
    pool = make_pool(2)
    pool.mark_invalid("key1")
    pool.mark_invalid("key2")
    with pytest.raises(NoKeyAvailable):
        pool.acquire()


def test_masked_key():
    pool = KeyPool([KeyEntry(key="sk-abcdef123456", label="")])
    state = pool.acquire()
    assert state.masked == "sk-a…3456"
    assert state.label_or_masked == "key-1"  # auto-generated label wins
    pool = KeyPool([KeyEntry(key="sk-abcdef123456", label="primary")])
    assert pool.acquire().label_or_masked == "primary"
