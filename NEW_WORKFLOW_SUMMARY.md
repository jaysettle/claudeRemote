# Test-Driven Implementation Workflow

## Overview

The interactive-discussion model now implements a **test-driven workflow** where:
1. **Claude** writes the implementation code
2. **Codex** writes tests for Claude's code
3. Tests are executed automatically
4. User decides next action based on test results

## New Workflow Stages

```
DISCUSSION → implement → CONSENSUS → yes →
IMPLEMENTATION (Planning) → approve →
CODE_GEN (Claude codes) →
TEST_GEN (Codex writes tests) →
TEST_RUN (Execute tests) →
POST_TEST (User decides) → deploy/fix/revise
```

## Key Changes from Previous Version

### Before:
- Both Claude and Codex generated code independently
- **Problem**: Second model's code overwrote first model's code
- No automated testing
- User deployed blindly without validation

### After:
- **Claude only** generates implementation code
- **Codex only** generates test code (sees Claude's implementation)
- Tests run automatically in isolated environment
- User gets clear pass/fail results before deployment

## Stage Details

### CODE_GEN Stage
- **Trigger**: User types "approve" after planning stage
- **Action**: Claude generates implementation code
- **Output**: Shows Claude's code, extracts files
- **Next**: Automatically moves to TEST_GEN

### TEST_GEN Stage
- **Action**: Codex receives Claude's code and generates tests
- **Context**: Codex sees both the plan and Claude's implementation
- **Output**: Shows test files
- **Next**: Automatically moves to TEST_RUN

### TEST_RUN Stage
- **Action**:
  - Creates temporary directory
  - Writes implementation files
  - Writes test files
  - Runs `pytest -v` with 30s timeout
- **Output**: Full pytest results (stdout + stderr)
- **Next**: Moves to POST_TEST with test results

### POST_TEST Stage

#### If Tests Pass ✅
**User commands:**
- `deploy` - Deploy the implementation
- `revise: <feedback>` - Make changes before deploying
- `cancel` - Abort implementation

**Prompt shown:**
```
✅ All tests passed!

[test output]

Ready to deploy!

Type "deploy" to apply changes, "revise: <feedback>" to modify code, or "cancel" to abort.
```

#### If Tests Fail ❌
**User commands:**
- `fix` - Claude analyzes failures and attempts automatic fix, then re-runs tests
- `deploy` - Deploy anyway (not recommended, shows warning)
- `revise: <guidance>` - Provide custom guidance for fixes
- `cancel` - Abort implementation

**Prompt shown:**
```
❌ Tests failed!

[test output]

Tests failed.

Type "fix" to have models fix the issues, "deploy" to deploy anyway (not recommended),
"revise: <feedback>" to provide guidance, or "cancel" to abort.
```

## Fix Command Details

When user types `fix`:
1. Claude receives:
   - Full test failure output
   - Original implementation code
   - Prompt: "Fix the code to pass the tests"
2. Claude generates fixed code
3. Tests re-run automatically with fixed code
4. User sees new test results
5. If still failing, can type `fix` again or provide custom guidance

## Deployment

Deployment logic is the same whether from IMPLEMENTATION stage or POST_TEST stage:
- Safety checks (path validation)
- Git rollback checkpoint (if in git repo)
- File backups before writing
- Writes all implementation files
- Health check validation
- Automatic rollback on error
- Success message shows backup location

## Example Session

```
User: "Create a Python function that calculates factorial"
→ Setup stage

User: "start"
→ Round 1 (Claude and Codex discuss approach)

User: "implement"
→ CONSENSUS stage (models propose what to build)

User: "yes"
→ IMPLEMENTATION stage (models create detailed plan)

User: "approve"
→ CODE_GEN: Claude writes factorial function
→ TEST_GEN: Codex writes tests (edge cases, 0!, negative numbers, etc.)
→ TEST_RUN: pytest runs

Result: ✅ Tests passed!

User: "deploy"
→ Writes /home/jay/projects/math/factorial.py
→ Success!
```

## Benefits

1. **No code conflicts** - Only one model writes implementation
2. **Automatic validation** - Tests catch bugs before deployment
3. **Iterative fixing** - Models can fix failures automatically
4. **User control** - Clear decision points with test evidence
5. **Collaborative strengths** - Claude codes, Codex tests (complementary roles)

## Test Environment

- Tests run in temporary isolated directory (`/tmp/discussion_test_*`)
- Uses `pytest` with verbose output
- 30-second timeout per test run
- Automatic cleanup after results captured
- No pollution of actual project directory until deployment

## Logging

New log traces:
- `code_gen.extracted` - Claude's code files captured
- `test_gen.extracted` - Codex's test files captured
- `test_run.start` - Test execution begins
- `test_run.write` - Files written to test directory
- `test_run.complete` - Test results (passed=true/false)
- `post_test.command` - User command in POST_TEST stage
- `post_test.fix` - Automatic fix attempt
- `post_test.deploy` - Deployment from POST_TEST
- `post_test.deploy_despite_failure` - Warning: deploying with failing tests

## Version

- **Dev Bridge Version**: 1.12.2-dev
- **Feature**: Test-driven implementation workflow
- **Date**: 2025-12-08
- **Branch**: Agentic-Team2

## Testing

To test this workflow:
1. Go to http://192.168.3.142:9001
2. Select model: `interactive-discussion`
3. Start a discussion about a simple feature
4. Type "implement" → "yes" → "approve"
5. Watch Claude code → Codex test → Auto-run tests
6. Make decision based on test results

## Future Enhancements

Potential improvements:
- Support for multiple test frameworks (unittest, nose, etc.)
- Test coverage metrics
- Performance benchmarks
- Language-specific test runners (Jest for JS, cargo test for Rust)
- Save test results to discussion history for export
- "test-first" mode (generate tests before implementation)
