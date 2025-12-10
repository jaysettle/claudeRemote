#!/usr/bin/env python3
"""
Pydantic Models for OpenAI-compatible API
Data structures for requests and responses
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ImageUrl(BaseModel):
    """Image URL in content parts"""
    url: str


class ContentPart(BaseModel):
    """Content part that can be text or image"""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    """Chat message with role and content"""
    role: str
    content: Union[str, List[ContentPart]]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None


class Model(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "bridge"


class ModelList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[Model]


# ============================================================================
# Discussion System Models
# ============================================================================

class DiscussionStage(Enum):
    """Discussion workflow stages"""
    SETUP = "setup"
    INITIAL = "initial"
    DISCUSSION = "discussion"
    ROUND1 = "round1"
    SYNTHESIS = "synthesis"
    CONSENSUS = "consensus"
    IMPLEMENTATION = "implementation"
    CODE_GEN = "code_gen"
    TEST_GEN = "test_gen"
    TEST_RUN = "test_run"
    POST_TEST = "post_test"
    COMPLETE = "complete"


@dataclass
class DiscussionState:
    """State management for interactive discussions"""
    chat_id: str
    stage: DiscussionStage = DiscussionStage.SETUP
    topic: str = ""
    models: List[str] = None
    mode: str = "collaborate"
    max_rounds: int = 999
    current_round: int = 0
    model_sessions: Dict[str, Optional[str]] = None
    discussion_history: List[Dict[str, str]] = None
    waiting_for_user: bool = False
    consensus_proposal: str = ""
    implementation_plan: str = ""
    implementation_code: Dict[str, str] = None
    test_code: Dict[str, str] = None
    implementation_approved: bool = False
    test_results: str = ""
    tests_passed: bool = False
    rollback_commit: str = ""

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


# ============================================================================
# Custom Exceptions
# ============================================================================

class SessionNotFoundError(Exception):
    """Raised when Claude CLI session is not found."""
    pass


class CodexSessionNotFoundError(Exception):
    """Raised when Codex CLI session is not found."""
    pass


class GeminiSessionNotFoundError(Exception):
    """Raised when Gemini CLI session is not found."""
    pass
