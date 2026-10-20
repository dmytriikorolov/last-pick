"""Run the shared match collector Choose the discovery method in TOML"""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.cli import ingest_main

if __name__ == "__main__":
    ingest_main()
