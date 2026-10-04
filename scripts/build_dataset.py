"""Compatibility entry point; reusable code lives in lastpick.legacy.dataset_csv."""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.legacy.dataset_csv import *

if __name__ == "__main__":
    main()
