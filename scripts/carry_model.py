"""Compatibility entry point; reusable code lives in lastpick.models.carry."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.models.carry import *
