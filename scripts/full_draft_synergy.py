"""Keep the original script command working
The implementation lives in lastpick.legacy.full_draft_model
"""

if __package__:
    from . import _bootstrap as _bootstrap
else:
    import _bootstrap as _bootstrap

from lastpick.legacy.full_draft_model import *
