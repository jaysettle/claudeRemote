#!/usr/bin/env python3
"""
Discussion State Management
Dataclasses and state storage for interactive discussions
"""

import time
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DiscussionStage(Enum):
    SETUP = "setup"
    INITIAL = "initial"
    DISCUSSION = "discussion"  # Multi-round interactive stage
    ROUND1 = "round1"
    SYNTHESIS = "synthesis"
    CONSENSUS = "consensus"  # Agree on what to implement
    IMPLEMENTATION = "implementation"  # Planning stage
    CODE_GEN = "code_gen"  # Claude generates code
    TEST_GEN = "test_gen"  # Codex generates tests
    TEST_RUN = "test_run"  # Execute tests and report
    POST_TEST = "post_test"  # User decides next action
    COMPLETE = "complete"

@dataclass
class DiscussionState:
    chat_id: str
    stage: DiscussionStage = DiscussionStage.SETUP
    topic: str = ""
    models: List[str] = None
    mode: str = "collaborate"  # collaborate or debate
    max_rounds: int = 999  # Effectively unlimited - user decides when to stop
    current_round: int = 0
    model_sessions: Dict[str, Optional[str]] = None
    discussion_history: List[Dict[str, str]] = None
    waiting_for_user: bool = False
    # Implementation mode fields
    consensus_proposal: str = ""  # What models think should be implemented
    implementation_plan: str = ""
    implementation_code: Dict[str, str] = None  # filename -> code content (Claude's)
    test_code: Dict[str, str] = None  # test filename -> test code (Codex's)
    implementation_approved: bool = False
    test_results: str = ""  # Test execution output
    tests_passed: bool = False
    rollback_commit: str = ""  # Git commit hash for rollback

    def __post_init__(self):
        if self.models is None:
            self.models = ["claude-cli", "codex-cli"]
        if self.test_code is None:
            self.test_code = {}
        if self.model_sessions is None:
            self.model_sessions = {}
        if self.discussion_history is None:
            self.discussion_history = []
        if self.implementation_code is None:
            self.implementation_code = {}

    def get_model_name(self, model_id: str) -> str:
        """Get friendly model name"""
        return model_id.replace("-cli", "").title()

    def add_response(self, model: str, response: str, stage: str):
        """Add a model response to the discussion history"""
        self.discussion_history.append({
            "stage": stage,
            "model": model,
            "response": response,
            "timestamp": time.time()
        })

# Global discussion states (simple in-memory storage for dev)
discussion_states: Dict[str, DiscussionState] = {}

def get_discussion_state(chat_id: str) -> Optional[DiscussionState]:
    """Get discussion state for a chat"""
    return discussion_states.get(chat_id)

def set_discussion_state(state: DiscussionState):
    """Save discussion state"""
    discussion_states[state.chat_id] = state

def clear_discussion_state(chat_id: str):
    """Clear discussion state"""
    if chat_id in discussion_states:
        del discussion_states[chat_id]



# Global discussion states (simple in-memory storage for dev)
discussion_states: Dict[str, DiscussionState] = {}


def get_discussion_state(chat_id: str) -> Optional[DiscussionState]:
    """Get discussion state for a chat"""
    return discussion_states.get(chat_id)


def set_discussion_state(state: DiscussionState):
    """Save discussion state"""
    discussion_states[state.chat_id] = state


def clear_discussion_state(chat_id: str):
    """Clear discussion state"""
    if chat_id in discussion_states:
        del discussion_states[chat_id]
