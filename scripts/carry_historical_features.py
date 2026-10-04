"""Compatibility entry point; reusable code lives in lastpick.features.carry."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.features.carry import *
