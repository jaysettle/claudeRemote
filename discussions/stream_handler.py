#!/usr/bin/env python3
"""
Interactive Discussion Stream Handler
Main handler for streaming interactive multi-model discussions
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Callable, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

from models import ChatCompletionRequest
from config import MCP_SYSTEM_PROMPT
from session_manager import get_session_id, set_session_id
from utils.helpers import emit_log as _emit_log, make_trace_logger

from .state import (
    DiscussionState, DiscussionStage,
    get_discussion_state, set_discussion_state, clear_discussion_state
)
from .prompts import (
    parse_discussion_intent, build_discussion_history_prompt, 
    format_discussion_prompt
)

logger = logging.getLogger(__name__)


# Forward declarations - these will be imported at runtime to avoid circular imports
def get_chat_id(request, messages):
    """Import and call get_chat_id from main module"""
    from claude_bridge import get_chat_id as _get_chat_id
    return _get_chat_id(request, messages)


async def call_model_prompt(model_id, prompt, session_id, history_prompt, chat_id, trace, **kwargs):
    """Import and call call_model_prompt from main module"""
    from claude_bridge import call_model_prompt as _call_model_prompt
    return await _call_model_prompt(model_id, prompt, session_id, history_prompt, chat_id, trace, **kwargs)


async def handle_interactive_discussion(request: Request, body: ChatCompletionRequest):
    """Handle interactive discussion mode - SAFE: only called for interactive-discussion model"""
    trace_id, trace = _make_trace_logger()
    chat_id = get_chat_id(request, body.messages)

    # Get or create discussion state
    state = get_discussion_state(chat_id)
    user_input = body.messages[-1].content if body.messages else ""

    logger.info(f"[{trace}] interactive_discussion.request | chat_id={chat_id} input='{user_input[:50]}...' has_state={state is not None} stage={state.stage.value if state else 'none'}")

    if body.stream:
        return StreamingResponse(
            stream_interactive_discussion(
                chat_id=chat_id,
                user_input=user_input,
                state=state,
                trace=trace,
            ),
            media_type="text/event-stream",
        )
    else:
        # Simple non-streaming response for testing
        result = "Interactive discussion mode active. Use streaming for full experience."
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "interactive-discussion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }]
        }

async def stream_interactive_discussion(
    chat_id: str,
    user_input: str,
    state: Optional[DiscussionState],
    trace: Optional[Callable] = None,
) -> AsyncGenerator[str, None]:
    """Stream interactive discussion - MINIMAL VERSION for testing"""

    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    def make_chunk(content: str, finish: bool = False):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "interactive-discussion",
            "choices": [{
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None
            }]
        }

    try:
        # Handle setup stage
        if not state or state.stage == DiscussionStage.SETUP:
            # Parse discussion request
            config = parse_discussion_intent(user_input)

            # Create new state
            state = DiscussionState(
                chat_id=chat_id,
                topic=config["topic"],
                models=config["models"],
                mode=config["mode"]
            )
            set_discussion_state(state)

            # Show setup
            model_names = [state.get_model_name(m) for m in state.models]
            yield f"data: {json.dumps(make_chunk(f'🎭 **Interactive Discussion Setup**\n\n'))}\n\n"
            # Truncate topic for display (first 200 chars or first line)
            topic_display = state.topic.split('\n')[0][:200]
            if len(state.topic) > 200 or '\n' in state.topic:
                topic_display += "..."
            yield f"data: {json.dumps(make_chunk(f'**Topic:** {topic_display}\n'))}\n\n"
            participants_text = ' vs '.join(model_names)
            yield f"data: {json.dumps(make_chunk(f'**Participants:** {participants_text}\n'))}\n\n"
            yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n\n'))}\n\n"

            # Move to initial stage
            state.stage = DiscussionStage.INITIAL
            set_discussion_state(state)

            yield f"data: {json.dumps(make_chunk('Type **\"start\"** to begin, or **\"cancel\"** to end.\n\n'))}\n\n"

        elif state.stage == DiscussionStage.INITIAL:
            user_command = user_input.lower().strip()

            if user_command == "cancel":
                clear_discussion_state(chat_id)
                yield f"data: {json.dumps(make_chunk('Discussion cancelled. 👋'))}\n\n"
            elif user_command == "start":
                # Run initial analysis
                yield f"data: {json.dumps(make_chunk(f'🎬 **Starting Discussion: {state.topic}**\n\n'))}\n\n"

                # Get responses from both models
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    # For debate mode, try to assign opposing positions from topic
                    debate_position = ""
                    if state.mode == "debate" and " or " in state.topic.lower():
                        # Extract "X or Y" pattern and assign positions
                        parts = state.topic.lower().split(" or ", 1)
                        if len(parts) == 2:
                            # First model gets first option, second model gets second option
                            option1 = parts[0].split()[-1] if parts[0].split() else parts[0]  # Last word before "or"
                            option2 = parts[1].split()[0] if parts[1].split() else parts[1]   # First word after "or"
                            debate_position = option1 if i == 0 else option2

                    # Show which position this model will argue for
                    if debate_position:
                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} analyzing... (arguing for: {debate_position})**\n'))}\n\n"
                    else:
                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} analyzing...**\n'))}\n\n"

                    prompt = format_discussion_prompt(state.topic, state.mode, is_initial=True, debate_position=debate_position)
                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    state.add_response(model, response, "initial")
                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                # Move to discussion rounds for interactive exchange
                state.current_round = 1
                state.stage = DiscussionStage.DISCUSSION
                set_discussion_state(state)

                if state.max_rounds > 1:
                    yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, provide your own guidance/question, **\"implement\"** to build an idea, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"
                else:
                    state.stage = DiscussionStage.SYNTHESIS
                    set_discussion_state(state)
                    yield f"data: {json.dumps(make_chunk('\n**Discussion complete!** Type **\"export\"** for summary.\n\n'))}\n\n"
            else:
                yield f"data: {json.dumps(make_chunk('Please type **\"start\"** or **\"cancel\"**.'))}\n\n"

        elif state.stage == DiscussionStage.DISCUSSION:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] discussion.command | chat_id={chat_id} cmd={user_command} stage={state.stage.value} round={state.current_round}")

            if user_command == "stop":
                state.stage = DiscussionStage.SYNTHESIS
                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('**Discussion ended.** Type **\"export\"** for summary.\n\n'))}\n\n"

            elif user_command == "export":
                # User wants to export immediately from DISCUSSION stage
                logger.info(f"[{trace}] discussion.export | chat_id={chat_id} round={state.current_round} entries={len(state.discussion_history)}")
                # Generate the export with summary directly
                yield f"data: {json.dumps(make_chunk('📋 **Discussion Export**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Topic:** {state.topic}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Rounds:** {state.current_round}\n\n'))}\n\n"

                yield f"data: {json.dumps(make_chunk('---\n\n## Round-by-Round Discussion\n\n'))}\n\n"

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    round_info = entry.get("stage", "")
                    yield f"data: {json.dumps(make_chunk(f'**{model_name}** ({round_info}):\n{response_text}\n\n'))}\n\n"

                # Generate AI summary using Claude
                yield f"data: {json.dumps(make_chunk('---\n\n## 🤖 AI-Generated Summary\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('*Analyzing discussion...*\n\n'))}\n\n"

                # Build summary prompt with full discussion history
                summary_parts = [f"Summarize this discussion about: {state.topic}\n\n"]
                summary_parts.append(f"Mode: {state.mode}\n")
                summary_parts.append(f"Total rounds: {state.current_round}\n\n")
                summary_parts.append("Discussion transcript:\n")

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    # Truncate very long responses for summary prompt
                    if len(response_text) > 1000:
                        response_text = response_text[:1000] + "... [truncated]"
                    summary_parts.append(f"{model_name}: {response_text}\n\n")

                summary_parts.append("""
Please provide:
1. **Key Points of Agreement**: What did both models agree on?
2. **Key Points of Disagreement**: Where did they differ?
3. **Evolution**: How did positions change across rounds?
4. **Conclusion**: What's the synthesized recommendation or outcome?

Keep it concise (3-5 paragraphs).""")

                summary_prompt = "".join(summary_parts)

                # Call Claude to generate summary
                try:
                    summary_response, _ = await call_model_prompt(
                        model_id="claude-cli",
                        prompt=summary_prompt,
                        session_id=None,  # Don't use discussion session
                        history_prompt=None,
                        chat_id=chat_id + "_summary",  # Different chat for summary
                        trace=trace,
                    )

                    yield f"data: {json.dumps(make_chunk(summary_response))}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps(make_chunk(f'*Summary generation failed: {str(e)}*'))}\n\n"

                yield f"data: {json.dumps(make_chunk('\n---\n\n*Export complete. Discussion ended.*\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            elif user_command == "implement" or user_input.lower().startswith("implement:"):
                # User wants to implement - first establish consensus
                logger.info(f"[{trace}] consensus.start | chat_id={chat_id}")

                state.stage = DiscussionStage.CONSENSUS
                state.consensus_proposal = ""
                set_discussion_state(state)

                yield f"data: {json.dumps(make_chunk(f'🤝 **Building Consensus**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('**Stage:** Identifying what to implement\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('Models are reviewing the discussion to propose what should be built...\n\n'))}\n\n"

                # Both models propose what they think should be implemented
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} proposing...**\n\n'))}\n\n"

                    # Build context from recent discussion
                    context_parts = []
                    if state.discussion_history:
                        context_parts.append("Recent discussion:\n")
                        for entry in state.discussion_history[-6:]:  # Last 6 entries
                            entry_model = state.get_model_name(entry["model"])
                            response = entry["response"][:400]  # First 400 chars
                            context_parts.append(f"{entry_model}: {response}...\n")
                    context_info = "\n".join(context_parts)

                    consensus_prompt = f"""{context_info}

The user wants to implement something from this discussion. Your task is to:

1. **Identify** what specific feature/project should be built based on the discussion
2. **Clarify** any ambiguities (what language? what's the goal? what scope?)
3. **Propose** a clear, concise statement of what will be implemented

Be specific. If the discussion was vague, ask clarifying questions.
If multiple options were discussed, state which one makes most sense and why.

Your proposal:"""

                    # Second model sees first model's proposal
                    if i == 1 and state.consensus_proposal:
                        consensus_prompt = f"""Your colleague proposed:\n\n{state.consensus_proposal}\n\n{consensus_prompt}

Do you agree? If not, what would you propose instead?"""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=consensus_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    # Store the proposal
                    if not state.consensus_proposal:
                        state.consensus_proposal = f"{model_name}:\n{response}\n\n"
                    else:
                        state.consensus_proposal += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Proposal complete!** Review what the models want to build.\n\nType **\"yes\"** to proceed with planning, **\"no: <clarification>\"** to provide more context, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "continue":
                # Check if we've hit max rounds
                if state.current_round >= state.max_rounds:
                    state.stage = DiscussionStage.SYNTHESIS
                    set_discussion_state(state)
                    yield f"data: {json.dumps(make_chunk(f'**Maximum rounds ({state.max_rounds}) reached.** Type **\"export\"** for summary.\n\n'))}\n\n"
                else:
                    # Run next round - models respond to each other (no user guidance)
                    state.current_round += 1
                    yield f"data: {json.dumps(make_chunk(f'\n🔄 **Round {state.current_round}** - Models responding to each other...\n\n'))}\n\n"

                    # Each model responds to the OTHER model's last response
                    for i, model in enumerate(state.models):
                        model_name = state.get_model_name(model)
                        other_model = state.models[1 - i]  # Get the other model
                        icon = "🔵" if i == 0 else "🟡"

                        # Find the other model's most recent response
                        other_response = None
                        for entry in reversed(state.discussion_history):
                            if entry["model"] == other_model:
                                other_response = entry["response"]
                                break

                        if not other_response:
                            continue

                        # Truncate other_response if too long (keep it under 2000 chars for context)
                        if len(other_response) > 2000:
                            other_response = other_response[:2000] + "\n\n[...response truncated for brevity...]"

                        yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} responding to {state.get_model_name(other_model)}...**\n'))}\n\n"

                        # Format prompt with the other model's response (no user guidance)
                        prompt = format_discussion_prompt(
                            state.topic,
                            state.mode,
                            is_initial=False,  # This is a follow-up
                            other_response=other_response
                        )

                        # Build history prompt for fallback if session is lost
                        history_prompt = build_discussion_history_prompt(state) + "\n\n" + prompt

                        response, session = await call_model_prompt(
                            model_id=model,
                            prompt=prompt,
                            session_id=get_session_id(model, chat_id),
                            history_prompt=history_prompt,
                            chat_id=chat_id,
                            trace=trace,
                        )

                        if session:
                            set_session_id(model, chat_id, session)

                        state.add_response(model, response, f"round_{state.current_round}")
                        yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                    # Offer to continue or stop
                    set_discussion_state(state)
                    if state.current_round >= state.max_rounds:
                        state.stage = DiscussionStage.SYNTHESIS
                        yield f"data: {json.dumps(make_chunk(f'\n**Discussion complete ({state.max_rounds} rounds).** Type **\"export\"** for summary.\n\n'))}\n\n"
                    else:
                        yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, provide guidance, **\"implement\"** to build an idea, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"
            else:
                # User provided free-form guidance instead of a command
                user_guidance = user_input  # Use the original input, not lowercased
                state.current_round += 1
                yield f"data: {json.dumps(make_chunk(f'\n💬 **Round {state.current_round}** - User guidance: "{user_guidance[:100]}{"..." if len(user_guidance) > 100 else ""}"\n\n'))}\n\n"

                # Both models respond to user's guidance
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    other_model = state.models[1 - i]
                    icon = "🔵" if i == 0 else "🟡"

                    # Find the other model's most recent response for context
                    other_response = None
                    for entry in reversed(state.discussion_history):
                        if entry["model"] == other_model:
                            other_response = entry["response"]
                            break

                    if not other_response:
                        other_response = "(No previous response from colleague)"

                    # Truncate other_response if too long
                    if len(other_response) > 2000:
                        other_response = other_response[:2000] + "\n\n[...response truncated for brevity...]"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} responding to user guidance...**\n'))}\n\n"

                    # Format prompt with user guidance AND other model's response
                    prompt = format_discussion_prompt(
                        state.topic,
                        state.mode,
                        is_initial=False,
                        other_response=other_response,
                        user_guidance=user_guidance  # Pass user's guidance
                    )

                    # Build history prompt for fallback if session is lost
                    history_prompt = build_discussion_history_prompt(state) + "\n\n" + prompt

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=history_prompt,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    state.add_response(model, response, f"round_{state.current_round}_user_guided")
                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                # Offer to continue or stop
                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk(f'\n**Round {state.current_round} complete.** Type **\"continue\"** for round {state.current_round + 1}, provide guidance, **\"implement\"** to build an idea, **\"export\"** for summary, or **\"stop\"** to end.\n\n'))}\n\n"

        elif state.stage == DiscussionStage.CONSENSUS:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] consensus.command | chat_id={chat_id} cmd={user_command}")

            if user_command == "yes":
                # User approved the consensus - move to planning
                logger.info(f"[{trace}] consensus.approved | chat_id={chat_id}")
                state.stage = DiscussionStage.IMPLEMENTATION
                state.implementation_plan = ""
                state.implementation_code = {}
                state.implementation_approved = False
                set_discussion_state(state)

                yield f"data: {json.dumps(make_chunk('✅ **Consensus reached!**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('🔧 **Implementation Mode**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('**Stage:** Planning\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('Models are creating a detailed implementation plan...\n\n'))}\n\n"

                # Extract what to implement from consensus
                feature_description = state.consensus_proposal

                # Detect project path from consensus/discussion
                project_path_hint = ""
                combined_text = state.consensus_proposal + "\n" + "\n".join([e["response"] for e in state.discussion_history])
                if "/home/jay/projects/" in combined_text:
                    import re
                    paths = re.findall(r'/home/jay/projects/[^\s]+', combined_text)
                    if paths:
                        project_path_hint = f"\nIMPORTANT: The project is located in: {paths[0]}\nAll file paths should be relative to this project directory.\n"

                # Both models create implementation plan
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} planning...**\n\n'))}\n\n"

                    planning_prompt = f"""Based on this consensus:\n\n{feature_description}{project_path_hint}

Create a detailed implementation plan. Include:
1. What files need to be created or modified
2. Specific code changes needed
3. Dependencies or prerequisites
4. How to test the implementation

Be concrete and specific about file paths and code.

Your plan:"""

                    if i == 1 and state.implementation_plan:
                        planning_prompt = f"""Your colleague's plan:\n\n{state.implementation_plan}\n\n{planning_prompt}

Review their plan and add your suggestions."""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=planning_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    if not state.implementation_plan:
                        state.implementation_plan = f"{model_name}:\n{response}\n\n"
                    else:
                        state.implementation_plan += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Plan complete!**\n\nType **\"approve\"** to generate code, **\"revise: <feedback>\"** to modify the plan, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_input.lower().startswith("no:"):
                # User wants to clarify - provide more context and try again
                clarification = user_input[3:].strip()
                logger.info(f"[{trace}] consensus.clarify | chat_id={chat_id} clarification='{clarification[:50]}'")

                yield f"data: {json.dumps(make_chunk(f'📝 **Clarification received:** {clarification}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('Models are revising their proposal...\n\n'))}\n\n"

                # Have models revise with clarification
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} revising...**\n\n'))}\n\n"

                    revision_prompt = f"""User clarification: "{clarification}"

Previous proposal:
{state.consensus_proposal}

Based on this clarification, propose what should be implemented. Be specific.

Your revised proposal:"""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=revision_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    if i == 0:
                        state.consensus_proposal = f"{model_name}:\n{response}\n\n"
                    else:
                        state.consensus_proposal += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Revised proposal complete!**\n\nType **\"yes\"** to proceed, **\"no: <more clarification>\"** to revise again, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "cancel":
                logger.info(f"[{trace}] consensus.cancelled | chat_id={chat_id}")
                yield f"data: {json.dumps(make_chunk('❌ **Consensus building cancelled.**\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            else:
                yield f"data: {json.dumps(make_chunk(f'**Unknown command:** "{user_command}"\n\nValid commands: **yes**, **no: <clarification>**, **cancel**\n\n'))}\n\n"

        elif state.stage == DiscussionStage.IMPLEMENTATION:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] implementation.command | chat_id={chat_id} cmd={user_command}")

            if user_command == "approve":
                # User approved the plan, Claude generates code
                logger.info(f"[{trace}] implementation.approved | chat_id={chat_id}")
                state.implementation_approved = True
                state.stage = DiscussionStage.CODE_GEN
                set_discussion_state(state)

                yield f"data: {json.dumps(make_chunk('✅ **Plan approved!**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('**Stage:** Code Generation\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('🔵 **Claude** is writing the implementation...\n\n'))}\n\n"

                # Only Claude generates code
                model = "claude-cli"

                # Create code generation prompt
                code_prompt = f"""Based on this implementation plan:

{state.implementation_plan}

Generate the actual code changes. For each file:
1. Specify the full file path
2. Show the complete code or the specific changes needed

Format your response as:
```
FILE: /path/to/file.py
[complete file content or clear instructions for changes]
```

Be precise and complete. Include all necessary files.

Your code:"""

                response, session = await call_model_prompt(
                    model_id=model,
                    prompt=code_prompt,
                    session_id=get_session_id(model, chat_id),
                    history_prompt=None,
                    chat_id=chat_id,
                    trace=trace,
                )

                if session:
                    set_session_id(model, chat_id, session)

                yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                # Extract file path and code from response
                import re
                file_matches = re.findall(r'FILE:\s*(.+?)```(.+?)```', response, re.DOTALL)
                for file_path, code_content in file_matches:
                    file_path = file_path.strip()
                    code_content = code_content.strip()
                    state.implementation_code[file_path] = code_content
                    logger.info(f"[{trace}] code_gen.extracted | file={file_path} chars={len(code_content)}")

                if not state.implementation_code:
                    yield f"data: {json.dumps(make_chunk('⚠️ **No code files were extracted.** Claude may need clearer instructions.\n\nType **\"revise: provide code in FILE: format\"** to try again, or **\"cancel\"** to abort.\n\n'))}\n\n"
                else:
                    # Move to TEST_GEN stage - Codex writes tests
                    state.stage = DiscussionStage.TEST_GEN
                    set_discussion_state(state)

                    yield f"data: {json.dumps(make_chunk(f'\n✅ **Code generation complete!**\n\n'))}\n\n"
                    yield f"data: {json.dumps(make_chunk(f'Files created by Claude:\n'))}\n\n"
                    for filepath in state.implementation_code.keys():
                        yield f"data: {json.dumps(make_chunk(f'- {filepath}\n'))}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'\n**Stage:** Test Generation\n\n'))}\n\n"
                    yield f"data: {json.dumps(make_chunk('🟡 **Codex** is writing tests for the implementation...\n\n'))}\n\n"

                    # Codex generates tests
                    test_model = "codex-cli"

                    # Build context: show Claude's code to Codex
                    code_context = "\n\n".join([f"FILE: {fp}\n```\n{code}\n```" for fp, code in state.implementation_code.items()])

                    test_prompt = f"""Claude has implemented this code:

{code_context}

Based on this implementation plan:
{state.implementation_plan}

Your task: Generate comprehensive tests for this code.

Format your response as:
```
FILE: /path/to/test_file.py
[complete test code]
```

Include:
- Unit tests for core functionality
- Edge cases and error handling
- Test to verify the basic requirement works

Your tests:"""

                    test_response, test_session = await call_model_prompt(
                        model_id=test_model,
                        prompt=test_prompt,
                        session_id=get_session_id(test_model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if test_session:
                        set_session_id(test_model, chat_id, test_session)

                    yield f"data: {json.dumps(make_chunk(f'{test_response}\n\n'))}\n\n"

                    # Extract test files
                    test_matches = re.findall(r'FILE:\s*(.+?)```(.+?)```', test_response, re.DOTALL)
                    for test_path, test_content in test_matches:
                        test_path = test_path.strip()
                        test_content = test_content.strip()
                        state.test_code[test_path] = test_content
                        logger.info(f"[{trace}] test_gen.extracted | file={test_path} chars={len(test_content)}")

                    if not state.test_code:
                        yield f"data: {json.dumps(make_chunk('\n⚠️ **No test files were extracted.** Proceeding without tests.\n\n'))}\n\n"
                        yield f"data: {json.dumps(make_chunk('Type **\"deploy\"** to apply changes without tests, or **\"cancel\"** to abort.\n\n'))}\n\n"
                    else:
                        yield f"data: {json.dumps(make_chunk(f'\n✅ **Test generation complete!**\n\n'))}\n\n"
                        yield f"data: {json.dumps(make_chunk(f'Test files created by Codex:\n'))}\n\n"
                        for test_path in state.test_code.keys():
                            yield f"data: {json.dumps(make_chunk(f'- {test_path}\n'))}\n\n"

                        # Move to TEST_RUN stage
                        state.stage = DiscussionStage.TEST_RUN
                        set_discussion_state(state)

                        yield f"data: {json.dumps(make_chunk(f'\n**Stage:** Running Tests\n\n'))}\n\n"

                        # Write files temporarily and run tests
                        import tempfile
                        from pathlib import Path

                        test_dir = Path(tempfile.mkdtemp(prefix="discussion_test_"))
                        logger.info(f"[{trace}] test_run.start | test_dir={test_dir}")

                        try:
                            # Write implementation files
                            for file_path, code_content in state.implementation_code.items():
                                target_file = test_dir / Path(file_path).name
                                target_file.write_text(code_content)
                                logger.info(f"[{trace}] test_run.write | file={target_file}")

                            # Write test files
                            for test_path, test_content in state.test_code.items():
                                target_file = test_dir / Path(test_path).name
                                target_file.write_text(test_content)
                                logger.info(f"[{trace}] test_run.write | file={target_file}")

                            # Run pytest
                            import subprocess
                            result = subprocess.run(
                                ["python", "-m", "pytest", "-v", str(test_dir)],
                                capture_output=True,
                                text=True,
                                timeout=30,
                                cwd=str(test_dir)
                            )

                            state.test_results = result.stdout + "\n" + result.stderr
                            state.tests_passed = (result.returncode == 0)
                            logger.info(f"[{trace}] test_run.complete | passed={state.tests_passed} returncode={result.returncode}")

                            if state.tests_passed:
                                yield f"data: {json.dumps(make_chunk('✅ **All tests passed!**\n\n'))}\n\n"
                            else:
                                yield f"data: {json.dumps(make_chunk('❌ **Tests failed!**\n\n'))}\n\n"

                            yield f"data: {json.dumps(make_chunk(f'```\n{state.test_results}\n```\n\n'))}\n\n"

                        except subprocess.TimeoutExpired:
                            state.test_results = "Tests timed out after 30 seconds"
                            state.tests_passed = False
                            yield f"data: {json.dumps(make_chunk('⏱️ **Tests timed out!**\n\n'))}\n\n"
                        except Exception as e:
                            state.test_results = f"Test execution error: {str(e)}"
                            state.tests_passed = False
                            yield f"data: {json.dumps(make_chunk(f'⚠️ **Test execution failed:** {str(e)}\n\n'))}\n\n"
                        finally:
                            # Clean up test directory
                            import shutil
                            shutil.rmtree(test_dir, ignore_errors=True)

                        # Move to POST_TEST stage
                        state.stage = DiscussionStage.POST_TEST
                        set_discussion_state(state)

                        if state.tests_passed:
                            yield f"data: {json.dumps(make_chunk('\n**Ready to deploy!**\n\nType **\"deploy\"** to apply changes, **\"revise: <feedback>\"** to modify code, or **\"cancel\"** to abort.\n\n'))}\n\n"
                        else:
                            yield f"data: {json.dumps(make_chunk('\n**Tests failed.**\n\nType **\"fix\"** to have models fix the issues, **\"deploy\"** to deploy anyway (not recommended), **\"revise: <feedback>\"** to provide guidance, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "deploy":
                # Deploy the code changes
                logger.info(f"[{trace}] implementation.deploy | chat_id={chat_id} files={len(state.implementation_code)}")

                if not state.implementation_code:
                    yield f"data: {json.dumps(make_chunk('⚠️ No code files captured to deploy. Try **"approve"** again.\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk('🚀 **Deploying changes...**\n\n'))}\n\n"

                    # Pre-flight safety check
                    unsafe_files = []
                    resolved_paths: Dict[str, Path] = {}
                    for filepath in state.implementation_code.keys():
                        resolved = Path(filepath).expanduser()
                        resolved_paths[filepath] = resolved
                        if not is_safe_path(resolved):
                            unsafe_files.append(str(resolved))

                    if unsafe_files:
                        warning = "\n".join([f"- {p}" for p in unsafe_files])
                        yield f"data: {json.dumps(make_chunk(f'⛔ Blocked unsafe paths. Adjust the target files and retry:\n{warning}\n\n'))}\n\n"
                    else:
                        git_root = None
                        for resolved in resolved_paths.values():
                            git_root = find_git_root(resolved)
                            if git_root:
                                break

                        rollback_commit = create_rollback_commit(git_root) if git_root else None
                        state.rollback_commit = rollback_commit or ""
                        set_discussion_state(state)

                        if rollback_commit:
                            yield f"data: {json.dumps(make_chunk(f'💾 Rollback checkpoint created at {rollback_commit}\n\n'))}\n\n"
                        elif git_root:
                            yield f"data: {json.dumps(make_chunk('⚠️ Git rollback skipped (dirty tree or commit failed). Using file backups only.\n\n'))}\n\n"
                        else:
                            yield f"data: {json.dumps(make_chunk('ℹ️ No git repository detected. Using file backups for rollback.\n\n'))}\n\n"

                        backup_dir = Path(tempfile.mkdtemp(prefix="implementation_backup_"))
                        created_files: List[Path] = []
                        write_error: Optional[str] = None

                        for filepath, code in state.implementation_code.items():
                            target = resolved_paths[filepath]
                            yield f"data: {json.dumps(make_chunk(f'✏️ Writing {target}\n'))}\n\n"
                            try:
                                written, created = write_file_with_backup(target, code, backup_dir)
                                if created:
                                    created_files.append(written)
                                yield f"data: {json.dumps(make_chunk(f'✅ Wrote {written}\n'))}\n\n"
                            except Exception as exc:
                                write_error = f"Failed to write {target}: {exc}"
                                logger.exception(write_error)
                                break

                        if write_error:
                            yield f"data: {json.dumps(make_chunk('⚠️ Error during write. Rolling back...\n\n'))}\n\n"
                            restored = restore_backups(backup_dir)
                            removed = delete_created_files(created_files)
                            rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                            yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk(write_error + '\n\n'))}\n\n"
                            clear_discussion_state(chat_id)
                        else:
                            # Health check after writes
                            health_ok, health_detail = check_service_health()
                            if not health_ok:
                                yield f"data: {json.dumps(make_chunk(f'❌ Health check failed: {health_detail}\nRolling back...\n\n'))}\n\n"
                                restored = restore_backups(backup_dir)
                                removed = delete_created_files(created_files)
                                rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                                yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                                clear_discussion_state(chat_id)
                            else:
                                yield f"data: {json.dumps(make_chunk(f'🩺 Health check passed: {health_detail}\n\n'))}\n\n"
                                yield f"data: {json.dumps(make_chunk(f'✅ **Implementation complete.** Backups stored at {backup_dir}.\n\n'))}\n\n"
                                clear_discussion_state(chat_id)

            elif user_input.lower().startswith("revise:"):
                # User wants to revise the plan
                feedback = user_input[7:].strip()
                logger.info(f"[{trace}] implementation.revise | chat_id={chat_id} feedback='{feedback[:50]}'")

                yield f"data: {json.dumps(make_chunk(f'🔄 **Revising plan based on feedback...**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Feedback:** {feedback}\n\n'))}\n\n"

                # Have models revise the plan
                for i, model in enumerate(state.models):
                    model_name = state.get_model_name(model)
                    icon = "🔵" if i == 0 else "🟡"

                    yield f"data: {json.dumps(make_chunk(f'{icon} **{model_name} revising...**\n\n'))}\n\n"

                    revision_prompt = f"""The user provided this feedback on the implementation plan:

"{feedback}"

Previous plan:
{state.implementation_plan}

Revise the plan based on this feedback. Address the user's concerns and improve the approach.

Your revised plan:"""

                    response, session = await call_model_prompt(
                        model_id=model,
                        prompt=revision_prompt,
                        session_id=get_session_id(model, chat_id),
                        history_prompt=None,
                        chat_id=chat_id,
                        trace=trace,
                    )

                    if session:
                        set_session_id(model, chat_id, session)

                    if i == 0:
                        state.implementation_plan = f"{model_name}:\n{response}\n\n"
                    else:
                        state.implementation_plan += f"{model_name}:\n{response}\n\n"

                    yield f"data: {json.dumps(make_chunk(f'{response}\n\n'))}\n\n"

                set_discussion_state(state)
                yield f"data: {json.dumps(make_chunk('\n**Revised plan complete!**\n\nType **\"approve\"** to proceed, **\"revise: <more feedback>\"** to revise again, or **\"cancel\"** to abort.\n\n'))}\n\n"

            elif user_command == "cancel":
                logger.info(f"[{trace}] implementation.cancelled | chat_id={chat_id}")
                yield f"data: {json.dumps(make_chunk('❌ **Implementation cancelled.**\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            else:
                yield f"data: {json.dumps(make_chunk(f'**Unknown command:** "{user_command}"\n\nValid commands: **approve**, **revise: <feedback>**, **deploy**, **cancel**\n\n'))}\n\n"

        elif state.stage == DiscussionStage.POST_TEST:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] post_test.command | chat_id={chat_id} cmd={user_command} passed={state.tests_passed}")

            if user_command == "deploy":
                # User wants to deploy (either tests passed or deploying despite failures)
                logger.info(f"[{trace}] post_test.deploy | chat_id={chat_id} files={len(state.implementation_code)} passed={state.tests_passed}")

                if not state.tests_passed:
                    logger.warning(f"[{trace}] post_test.deploy_despite_failure | chat_id={chat_id}")
                    yield f"data: {json.dumps(make_chunk('⚠️ **Deploying despite test failures...**\n\n'))}\n\n"

                if not state.implementation_code:
                    yield f"data: {json.dumps(make_chunk('⚠️ No code files to deploy.\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk('🚀 **Deploying changes...**\n\n'))}\n\n"

                    # Pre-flight safety check
                    from pathlib import Path
                    import tempfile
                    from typing import Dict, List, Optional

                    unsafe_files = []
                    resolved_paths: Dict[str, Path] = {}
                    for filepath in state.implementation_code.keys():
                        resolved = Path(filepath).expanduser()
                        resolved_paths[filepath] = resolved
                        if not is_safe_path(resolved):
                            unsafe_files.append(str(resolved))

                    if unsafe_files:
                        warning = "\n".join([f"- {p}" for p in unsafe_files])
                        yield f"data: {json.dumps(make_chunk(f'⛔ Blocked unsafe paths:\n{warning}\n\n'))}\n\n"
                    else:
                        git_root = None
                        for resolved in resolved_paths.values():
                            git_root = find_git_root(resolved)
                            if git_root:
                                break

                        rollback_commit = create_rollback_commit(git_root) if git_root else None
                        state.rollback_commit = rollback_commit or ""
                        set_discussion_state(state)

                        if rollback_commit:
                            yield f"data: {json.dumps(make_chunk(f'💾 Rollback checkpoint: {rollback_commit}\n\n'))}\n\n"
                        elif git_root:
                            yield f"data: {json.dumps(make_chunk('⚠️ Git rollback skipped (dirty tree). Using file backups only.\n\n'))}\n\n"
                        else:
                            yield f"data: {json.dumps(make_chunk('ℹ️ No git repository detected. Using file backups.\n\n'))}\n\n"

                        backup_dir = Path(tempfile.mkdtemp(prefix="implementation_backup_"))
                        created_files: List[Path] = []
                        write_error: Optional[str] = None

                        for filepath, code in state.implementation_code.items():
                            target = resolved_paths[filepath]
                            yield f"data: {json.dumps(make_chunk(f'✏️ Writing {target}\n'))}\n\n"
                            try:
                                written, created = write_file_with_backup(target, code, backup_dir)
                                if created:
                                    created_files.append(written)
                                yield f"data: {json.dumps(make_chunk(f'✅ Wrote {written}\n'))}\n\n"
                            except Exception as exc:
                                write_error = f"Failed to write {target}: {exc}"
                                logger.exception(write_error)
                                break

                        if write_error:
                            yield f"data: {json.dumps(make_chunk('⚠️ Error during write. Rolling back...\n\n'))}\n\n"
                            restored = restore_backups(backup_dir)
                            removed = delete_created_files(created_files)
                            rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                            yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk(write_error + '\n\n'))}\n\n"
                            clear_discussion_state(chat_id)
                        else:
                            # Health check after writes
                            health_ok, health_detail = check_service_health()
                            if not health_ok:
                                yield f"data: {json.dumps(make_chunk(f'❌ Health check failed: {health_detail}\nRolling back...\n\n'))}\n\n"
                                restored = restore_backups(backup_dir)
                                removed = delete_created_files(created_files)
                                rollback_ok = reset_to_commit(git_root, state.rollback_commit) if state.rollback_commit else False
                                yield f"data: {json.dumps(make_chunk(f'Rolled back {len(restored)} files; removed {len(removed)} new files. Git rollback: {"ok" if rollback_ok else "skipped"}.\n\n'))}\n\n"
                                clear_discussion_state(chat_id)
                            else:
                                yield f"data: {json.dumps(make_chunk(f'🩺 Health check passed: {health_detail}\n\n'))}\n\n"

                                if state.tests_passed:
                                    yield f"data: {json.dumps(make_chunk(f'✅ **Implementation complete!** All tests passed. Backups at {backup_dir}.\n\n'))}\n\n"
                                else:
                                    yield f"data: {json.dumps(make_chunk(f'✅ **Implementation complete** (deployed despite test failures). Backups at {backup_dir}.\n\n'))}\n\n"

                                clear_discussion_state(chat_id)

            elif user_command == "fix":
                # Models attempt to fix test failures
                logger.info(f"[{trace}] post_test.fix | chat_id={chat_id}")
                yield f"data: {json.dumps(make_chunk('🔧 **Attempting to fix test failures...**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('🔵 **Claude** analyzing failures...\n\n'))}\n\n"

                fix_prompt = f"""The implementation has test failures:

Test Results:
```
{state.test_results}
```

Your original code:
{chr(10).join([f"FILE: {fp}{chr(10)}```{chr(10)}{code}{chr(10)}```" for fp, code in state.implementation_code.items()])}

Fix the code to pass the tests. Provide corrected code in the same FILE: format.

Your fixed code:"""

                fix_response, fix_session = await call_model_prompt(
                    model_id="claude-cli",
                    prompt=fix_prompt,
                    session_id=get_session_id("claude-cli", chat_id),
                    history_prompt=None,
                    chat_id=chat_id,
                    trace=trace,
                )

                if fix_session:
                    set_session_id("claude-cli", chat_id, fix_session)

                yield f"data: {json.dumps(make_chunk(f'{fix_response}\n\n'))}\n\n"

                # Extract fixed code
                import re
                fix_matches = re.findall(r'FILE:\s*(.+?)```(.+?)```', fix_response, re.DOTALL)
                if fix_matches:
                    for file_path, code_content in fix_matches:
                        file_path = file_path.strip()
                        code_content = code_content.strip()
                        state.implementation_code[file_path] = code_content
                        logger.info(f"[{trace}] post_test.fix_extracted | file={file_path} chars={len(code_content)}")

                    # Re-run tests with fixed code
                    yield f"data: {json.dumps(make_chunk('\n🔄 **Re-running tests with fixes...**\n\n'))}\n\n"

                    import tempfile
                    from pathlib import Path
                    import shutil

                    test_dir = Path(tempfile.mkdtemp(prefix="discussion_test_retry_"))

                    try:
                        # Write fixed implementation files
                        for file_path, code_content in state.implementation_code.items():
                            target_file = test_dir / Path(file_path).name
                            target_file.write_text(code_content)

                        # Write test files
                        for test_path, test_content in state.test_code.items():
                            target_file = test_dir / Path(test_path).name
                            target_file.write_text(test_content)

                        # Run pytest again
                        import subprocess
                        result = subprocess.run(
                            ["python", "-m", "pytest", "-v", str(test_dir)],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            cwd=str(test_dir)
                        )

                        state.test_results = result.stdout + "\n" + result.stderr
                        state.tests_passed = (result.returncode == 0)

                        if state.tests_passed:
                            yield f"data: {json.dumps(make_chunk('✅ **Tests now pass!**\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk(f'```\n{state.test_results}\n```\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk('\nType **\"deploy\"** to apply changes, or **\"cancel\"** to abort.\n\n'))}\n\n"
                        else:
                            yield f"data: {json.dumps(make_chunk('❌ **Tests still failing.**\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk(f'```\n{state.test_results}\n```\n\n'))}\n\n"
                            yield f"data: {json.dumps(make_chunk('\nType **\"fix\"** to try again, **\"revise: <guidance>\"** for custom fixes, **\"deploy\"** to deploy anyway, or **\"cancel\"** to abort.\n\n'))}\n\n"

                    except Exception as e:
                        yield f"data: {json.dumps(make_chunk(f'⚠️ **Fix test failed:** {str(e)}\n\n'))}\n\n"
                    finally:
                        shutil.rmtree(test_dir, ignore_errors=True)

                    set_discussion_state(state)
                else:
                    yield f"data: {json.dumps(make_chunk('⚠️ **No fixed code extracted.** Try **\"revise: <specific guidance>\"** instead.\n\n'))}\n\n"

            elif user_input.lower().startswith("revise:"):
                # User provides custom guidance for fixes
                feedback = user_input[7:].strip()
                logger.info(f"[{trace}] post_test.revise | chat_id={chat_id} feedback='{feedback[:50]}'")

                yield f"data: {json.dumps(make_chunk(f'📝 **Custom revision:** {feedback}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('🔵 **Claude** revising based on feedback...\n\n'))}\n\n"

                revise_prompt = f"""User feedback: "{feedback}"

Test Results:
```
{state.test_results}
```

Current code:
{chr(10).join([f"FILE: {fp}{chr(10)}```{chr(10)}{code}{chr(10)}```" for fp, code in state.implementation_code.items()])}

Revise the code according to user feedback. Provide updated code in FILE: format.

Your revised code:"""

                revise_response, revise_session = await call_model_prompt(
                    model_id="claude-cli",
                    prompt=revise_prompt,
                    session_id=get_session_id("claude-cli", chat_id),
                    history_prompt=None,
                    chat_id=chat_id,
                    trace=trace,
                )

                if revise_session:
                    set_session_id("claude-cli", chat_id, revise_session)

                yield f"data: {json.dumps(make_chunk(f'{revise_response}\n\n'))}\n\n"

                # Extract revised code (same logic as fix)
                import re
                revise_matches = re.findall(r'FILE:\s*(.+?)```(.+?)```', revise_response, re.DOTALL)
                if revise_matches:
                    for file_path, code_content in revise_matches:
                        file_path = file_path.strip()
                        code_content = code_content.strip()
                        state.implementation_code[file_path] = code_content

                    set_discussion_state(state)
                    yield f"data: {json.dumps(make_chunk('\n**Revision complete.** Type **\"deploy\"** to apply changes, or **\"cancel\"** to abort.\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk('⚠️ **No revised code extracted.**\n\n'))}\n\n"

            elif user_command == "cancel":
                logger.info(f"[{trace}] post_test.cancelled | chat_id={chat_id}")
                yield f"data: {json.dumps(make_chunk('❌ **Implementation cancelled.**\n\n'))}\n\n"
                clear_discussion_state(chat_id)

            else:
                if state.tests_passed:
                    yield f"data: {json.dumps(make_chunk(f'**Unknown command:** "{user_command}"\n\nValid commands: **deploy**, **revise: <feedback>**, **cancel**\n\n'))}\n\n"
                else:
                    yield f"data: {json.dumps(make_chunk(f'**Unknown command:** "{user_command}"\n\nValid commands: **fix**, **deploy**, **revise: <feedback>**, **cancel**\n\n'))}\n\n"

        elif state.stage == DiscussionStage.SYNTHESIS:
            user_command = user_input.lower().strip()
            logger.info(f"[{trace}] synthesis.command | chat_id={chat_id} cmd={user_command} stage={state.stage.value}")

            if user_command == "export":
                # Export with AI-generated summary
                logger.info(f"[{trace}] synthesis.export | chat_id={chat_id} round={state.current_round} entries={len(state.discussion_history)}")
                yield f"data: {json.dumps(make_chunk('📋 **Discussion Export**\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Topic:** {state.topic}\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Mode:** {state.mode.title()}\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk(f'**Rounds:** {state.current_round}\n\n'))}\n\n"

                yield f"data: {json.dumps(make_chunk('---\n\n## Round-by-Round Discussion\n\n'))}\n\n"

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    round_info = entry.get("stage", "")
                    yield f"data: {json.dumps(make_chunk(f'**{model_name}** ({round_info}):\n{response_text}\n\n'))}\n\n"

                # Generate AI summary using Claude
                yield f"data: {json.dumps(make_chunk('---\n\n## 🤖 AI-Generated Summary\n\n'))}\n\n"
                yield f"data: {json.dumps(make_chunk('*Analyzing discussion...*\n\n'))}\n\n"

                # Build summary prompt with full discussion history
                summary_parts = [f"Summarize this discussion about: {state.topic}\n\n"]
                summary_parts.append(f"Mode: {state.mode}\n")
                summary_parts.append(f"Total rounds: {state.current_round}\n\n")
                summary_parts.append("Discussion transcript:\n")

                for entry in state.discussion_history:
                    model_name = state.get_model_name(entry["model"])
                    response_text = entry["response"]
                    # Truncate very long responses for summary prompt
                    if len(response_text) > 1000:
                        response_text = response_text[:1000] + "... [truncated]"
                    summary_parts.append(f"{model_name}: {response_text}\n\n")

                summary_parts.append("""
Please provide:
1. **Key Points of Agreement**: What did both models agree on?
2. **Key Points of Disagreement**: Where did they differ?
3. **Evolution**: How did positions change across rounds?
4. **Conclusion**: What's the synthesized recommendation or outcome?

Keep it concise (3-5 paragraphs).""")

                summary_prompt = "".join(summary_parts)

                # Call Claude to generate summary
                try:
                    summary_response, _ = await call_model_prompt(
                        model_id="claude-cli",
                        prompt=summary_prompt,
                        session_id=None,  # Don't use discussion session
                        history_prompt=None,
                        chat_id=chat_id + "_summary",  # Different chat for summary
                        trace=trace,
                    )

                    yield f"data: {json.dumps(make_chunk(summary_response))}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps(make_chunk(f'*Summary generation failed: {str(e)}*'))}\n\n"

                yield f"data: {json.dumps(make_chunk('\n---\n\n*Export complete. Discussion ended.*\n\n'))}\n\n"
                clear_discussion_state(chat_id)
            else:
                # Unexpected input in SYNTHESIS stage
                logger.warning(f"[{trace}] synthesis.unexpected | chat_id={chat_id} input='{user_command}' expected='export'")
                yield f"data: {json.dumps(make_chunk(f'**Invalid command in synthesis stage:** "{user_command}"\n\nPlease type exactly **\"export\"** to generate summary.\n\n'))}\n\n"

    except Exception as e:
        error_msg = f"Discussion error: {str(e)}"
        _emit_log(trace, "interactive_discussion.error", str(e), level=logging.ERROR)
        yield f"data: {json.dumps(make_chunk(error_msg))}\n\n"

    yield f"data: {json.dumps(make_chunk('', finish=True))}\n\n"
    yield "data: [DONE]\n\n"


