#!/usr/bin/env python3
"""
Discussion Prompt Formatting
Handles prompt generation for different discussion stages and modes
"""

import logging
from typing import Any, Dict

from .state import DiscussionState

logger = logging.getLogger(__name__)


def parse_discussion_intent(user_input: str) -> Dict[str, Any]:
    """Parse user intent for discussion configuration"""
    config = {
        "topic": user_input,
        "models": ["claude-cli", "codex-cli"],
        "rounds": 2,
        "mode": "collaborate"
    }

    input_lower = user_input.lower()

    # Detect mode
    if any(word in input_lower for word in ["debate", "argue", "disagree", "vs", "versus"]):
        config["mode"] = "debate"

    # Detect model preferences
    if "gemini" in input_lower:
        if "codex" in input_lower:
            config["models"] = ["gemini-cli", "codex-cli"]
        else:
            config["models"] = ["claude-cli", "gemini-cli"]

    # Extract core topic - only strip known command words if they appear as the FIRST word
    topic = user_input
    first_word_raw = input_lower.strip().split()[0] if input_lower.strip() else ""
    # Strip common punctuation from first word for matching (debate: → debate)
    first_word = first_word_raw.rstrip(':,;.!?')

    # Known command words that should be stripped from the beginning
    command_words = ["discuss", "debate", "analyze", "compare"]

    if first_word in command_words:
        # Remove the first word and use the rest as topic
        words = user_input.strip().split(None, 1)
        topic = words[1] if len(words) > 1 else user_input
    else:
        # Use entire input as topic
        topic = user_input

    logger.info(f"parse_discussion_intent | first_word='{first_word}' | topic='{topic[:100]}...'")

    config["topic"] = topic if topic else user_input
    return config

def build_discussion_history_prompt(state: DiscussionState) -> str:
    """Build a history prompt with full discussion context for session fallback"""
    history_parts = []
    history_parts.append(f"[Original Topic]\n{state.topic}\n")
    history_parts.append(f"\n[Discussion History - {state.current_round} rounds]\n")

    for entry in state.discussion_history[-10:]:  # Last 10 entries
        model_name = state.get_model_name(entry["model"])
        response = entry["response"]
        if len(response) > 800:
            response = response[:800] + "... [truncated]"
        history_parts.append(f"\n{model_name}: {response}\n")

    return "\n".join(history_parts)

def format_discussion_prompt(topic: str, mode: str, is_initial: bool = True, other_response: str = "", user_guidance: str = "", debate_position: str = "") -> str:
    """Format prompts for discussion"""
    if is_initial:
        if mode == "debate":
            position_instruction = f"\n\nYou must argue FOR: {debate_position}" if debate_position else "\n\nTake a clear position and provide your strongest arguments."
            return f"""You are participating in a debate about: "{topic}"{position_instruction} Be intellectually rigorous but respectful.

Your analysis:"""
        else:
            return f"""You are collaborating on analyzing: "{topic}"

Provide your expertise and perspective. Focus on constructive analysis and suggestions.

Your analysis:"""
    else:
        # Response to other model - ENHANCED for deeper engagement
        # If user provided guidance, incorporate it into the prompt
        if user_guidance:
            if mode == "debate":
                return f"""The user has provided this guidance: "{user_guidance}"

Your colleague's previous response:
"{other_response}"

Now respond to BOTH the user's guidance AND your colleague's points:
- Address the user's question or direction
- Challenge or support your colleague's relevant points
- Present your perspective on the user's guidance

Your response:"""
            else:
                return f"""The user has provided this guidance: "{user_guidance}"

Your colleague's previous response:
"{other_response}"

Please respond to BOTH the user's guidance AND build on your colleague's analysis:
- Address the user's question or direction
- Integrate your colleague's relevant points
- Add your own perspective and expertise

Your response:"""
        else:
            # No user guidance - models respond to each other
            if mode == "debate":
                return f"""Your colleague just argued:

"{other_response}"

Now respond to their points:
- Challenge any claims you disagree with (provide reasoning)
- Acknowledge points where they're correct
- Present counter-arguments or alternative perspectives
- Ask clarifying questions if needed

Your rebuttal:"""
            else:
                return f"""Your colleague shared this analysis:

"{other_response}"

Build on their contribution:
- Identify points you agree with and why
- Add complementary perspectives they may have missed
- Address any gaps or concerns you see
- Suggest how to combine both viewpoints

Your response:"""
