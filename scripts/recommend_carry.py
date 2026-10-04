"""Compatibility entry point; reusable code lives in lastpick.recommender.carry."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.recommender.carry import *
