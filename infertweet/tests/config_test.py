# Copyright (C) 2013 Wesley Baugh
import os

from nose.tools import assert_raises

from infertweet.config import get_config


TEST_FNAME = '__nose__'


def test_get_config_none():
    assert_raises(OSError, os.remove, TEST_FNAME)
    result = get_config(fname=TEST_FNAME, create=False, exit=False)
    assert result is None


def test_get_config_create():
    get_config(fname=TEST_FNAME, create=True, exit=False)
    assert get_config(fname=TEST_FNAME, create=False, exit=False)
    os.remove(TEST_FNAME)


def test_get_config_exit():
    assert_raises(SystemExit, get_config,
                  fname=TEST_FNAME, create=False, exit=True)
    assert_raises(OSError, os.remove, TEST_FNAME)


def test_get_config_create_exit():
    assert_raises(SystemExit, get_config,
                  fname=TEST_FNAME, create=True, exit=True)
    os.remove(TEST_FNAME)
