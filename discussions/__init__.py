"""Interactive discussion module for multi-model collaboration"""
from .state import (
    DiscussionState, DiscussionStage, 
    get_discussion_state, set_discussion_state, clear_discussion_state
)
from .prompts import (
    parse_discussion_intent, build_discussion_history_prompt,
    format_discussion_prompt
)
from .stream_handler import (
    handle_interactive_discussion, stream_interactive_discussion
)

__all__ = [
    'DiscussionState', 'DiscussionStage',
    'get_discussion_state', 'set_discussion_state', 'clear_discussion_state',
    'parse_discussion_intent', 'build_discussion_history_prompt', 'format_discussion_prompt',
    'handle_interactive_discussion', 'stream_interactive_discussion'
]
