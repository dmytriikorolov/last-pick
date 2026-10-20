"""Keep the original script command working
The implementation lives in lastpick.legacy.hero_lookup
"""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.legacy.hero_lookup import *
