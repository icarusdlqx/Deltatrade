from __future__ import annotations
from .llm_client import smoke_test
from .agents_patch import apply as patch_agents
from .pretrade_guard import must_have_recent_event_assessment

def apply():
    patch_agents()
    smoke_test()
    _ = must_have_recent_event_assessment  # ensure guard import is kept for runtime users
    return True
