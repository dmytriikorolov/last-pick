"""Compatibility entry point; reusable code lives in lastpick.models.training."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.models.training import *

if __name__ == "__main__":
    main()
