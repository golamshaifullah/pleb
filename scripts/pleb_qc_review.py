#!/usr/bin/env python3
"""Compatibility launcher; installed users can run ``pleb review``."""

from pleb import review_app as _app


def __getattr__(name):
    return getattr(_app, name)


if __name__ == "__main__":
    _app.main()
