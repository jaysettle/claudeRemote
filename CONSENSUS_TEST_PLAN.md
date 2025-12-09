# CONSENSUS Stage Test Plan

## Test Environment
- **Open WebUI Dev**: http://192.168.3.142:9001
- **Model**: `interactive-discussion`
- **Feature**: CONSENSUS stage before implementation planning

## Test Scenario 1: Happy Path (React vs Vanilla JS)

### Step 1: Start Discussion
**Input:** `Should we build a simple calculator web app using React or vanilla JavaScript?`

**Expected Output:**
```
🎭 Interactive Discussion Setup

Topic: Should we build a simple calculator web app using React or vanilla JavaScript?
Participants: Claude vs Codex
Mode: Collaborate

Type "start" to begin, or "cancel" to end.
```

### Step 2: Begin Discussion
**Input:** `start`

**Expected Output:**
- 🎬 Starting Discussion...
- 🔵 Claude analyzing...
- [Claude's analysis of React vs Vanilla JS]
- 🟡 Codex analyzing...
- [Codex's analysis]
- **Round 1 complete.** Type "continue" for round 2, provide your own guidance/question, **"implement"** to build an idea, "export" for summary, or "stop" to end.

### Step 3: Continue Discussion
**Input:** `continue`

**Expected Output:**
- 🔄 Round 2 - Models responding to each other...
- [Claude responds to Codex's points]
- [Codex responds to Claude's points]
- **Round 2 complete.** Type "continue"... [same options]

### Step 4: Trigger Consensus
**Input:** `implement`

**Expected Output:**
```
🤝 Building Consensus

Stage: Identifying what to implement

Models are reviewing the discussion to propose what should be built...

🔵 Claude proposing...
[Claude's proposal of what to build based on discussion]

🟡 Codex proposing...
[Codex's proposal, possibly agreeing or suggesting alternative]

Proposal complete! Review what the models want to build.

Type "yes" to proceed with planning, "no: <clarification>" to provide more context, or "cancel" to abort.
```

**Verification Points:**
- ✅ Both models reference the React vs Vanilla JS discussion
- ✅ Proposals are specific (not vague "implement the idea discussed")
- ✅ Clear next step options shown

### Step 5: Approve Consensus
**Input:** `yes`

**Expected Output:**
```
✅ Consensus reached!

🔧 Implementation Mode

Stage: Planning

Models are creating a detailed implementation plan...

🔵 Claude planning...
[Detailed plan with files, dependencies, etc.]

🟡 Codex planning...
[Codex's plan or additions to Claude's plan]

Plan complete!

Type "approve" to generate code, "revise: <feedback>" to modify the plan, or "cancel" to abort.
```

**Verification Points:**
- ✅ Plan includes specific files to create/modify
- ✅ Plan references the consensus (React or Vanilla JS choice)
- ✅ Clear next steps shown

### Step 6: Approve Plan
**Input:** `approve`

**Expected Output:**
```
✅ Plan approved!

Stage: Code Generation

Models are generating code...

🔵 Claude coding...
[Code in FILE: format]

🟡 Codex coding...
[Code in FILE: format]

Code generation complete!

Files to modify: N
- /path/to/file1
- /path/to/file2

Type "deploy" to apply changes, or "cancel" to abort.
```

**Verification Points:**
- ✅ Code follows FILE: format
- ✅ Code matches the approved plan
- ✅ File paths are appropriate (not /home/jay/claude-cli-bridge-dev)

### Step 7: Cancel (Safety Test)
**Input:** `cancel`

**Expected Output:**
```
❌ Implementation cancelled.
```

## Test Scenario 2: Clarification Flow

### Steps 1-4: Same as Scenario 1
(Get to CONSENSUS stage)

### Step 5: Request Clarification
**Input:** `no: Make it a scientific calculator with trig functions`

**Expected Output:**
```
📝 Clarification received: Make it a scientific calculator with trig functions

Models are revising their proposal...

🔵 Claude revising...
[Updated proposal incorporating trig functions]

🟡 Codex revising...
[Updated proposal]

Revised proposal complete!

Type "yes" to proceed, "no: <more clarification>" to revise again, or "cancel" to abort.
```

**Verification Points:**
- ✅ Models incorporate the clarification
- ✅ Proposals now mention scientific/trig features
- ✅ Can iterate multiple times if needed

## Test Scenario 3: Project Path Detection

### Step 1: Discussion with Path
**Input:** `Let's create a hello world project in /home/jay/projects/hello-test`

**Steps 2-3:** Start and continue discussion

### Step 4-5: Implement and approve consensus
**Input:** `implement` → `yes`

**Expected in Planning Stage:**
```
IMPORTANT: The project is located in: /home/jay/projects/hello-test
All file paths should be relative to this project directory.
```

**Verification Points:**
- ✅ Planning prompts include project path hint
- ✅ Generated file paths are relative to /home/jay/projects/hello-test
- ✅ Not trying to modify /home/jay/claude-cli-bridge-dev

## Test Scenario 4: Invalid Commands

### At CONSENSUS Stage:
**Input:** `continue`

**Expected:**
```
Unknown command: "continue"

Valid commands: yes, no: <clarification>, cancel
```

### At IMPLEMENTATION (Planning) Stage:
**Input:** `start`

**Expected:**
```
Unknown command: "start"

Valid commands: approve, revise: <feedback>, deploy, cancel
```

## Success Criteria

✅ All stage transitions show clear next steps
✅ CONSENSUS stage catches misunderstandings before planning
✅ Models propose specific implementations (not "the thing discussed")
✅ Clarification loop works (no: <text>)
✅ Project path detection prevents wrong directory modifications
✅ Invalid commands show helpful error messages
✅ Cancel works at each stage
✅ State persists across requests (same chat)

## Known Issues to Watch For

⚠️ **State not persisting**: If typing "start" creates new setup, chat_id isn't being preserved
⚠️ **Models confused**: If proposals are vague, consensus prompt needs improvement
⚠️ **Wrong directory**: If paths point to claude-cli-bridge-dev, path detection failed
⚠️ **Missing prompts**: If any stage doesn't show next steps, that's a bug

## Debugging Commands

```bash
# View live logs
ssh jay@192.168.3.142
echo 'lowvo' | sudo -S journalctl -u claude-bridge-dev.service -f

# Check session storage
curl http://192.168.3.142:9000/sessions

# Health check
curl http://192.168.3.142:9000/
```

## Quick Test (Minimal)

If short on time, test this flow:
1. Start discussion about React vs Vanilla calculator
2. Type "start"
3. Type "implement" (skip continue rounds)
4. Verify CONSENSUS stage appears with clear proposals
5. Type "yes"
6. Verify IMPLEMENTATION planning stage
7. Type "cancel"

This tests the core CONSENSUS workflow without full implementation.
