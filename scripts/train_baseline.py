"""Compatibility entry point; reusable code lives in lastpick.legacy.logistic."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.legacy.logistic import *

if __name__ == "__main__":
    main()
