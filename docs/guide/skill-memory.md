# Skill Memory (`MemoryType.SKILL`)

`MemoryType.SKILL` is the memory category for **procedural knowledge**: a named procedure an agent can look up and follow. A skill stores the description, ordered steps, decision rules, and an optional tool schema in one atom, alongside the same embedding and knowledge-graph triples as any other memory.

Skills live in the database. EpochDB does not write them out as Markdown files.

---

## Where it sits among memory types

`MemoryType` is defined on every `UnifiedMemoryAtom`. The value is also copied onto the public `Memory` object as `memory.memory_type`.

| Value | Enum | What it stores |
| --- | --- | --- |
| `general` | `MemoryType.GENERAL` | Default. Facts and notes with no more specific category. |
| `episodic` | `MemoryType.EPISODIC` | Conversational context that should persist across sessions. |
| `profile` | `MemoryType.PROFILE` | Long-term user facts and preferences. |
| `working` | `MemoryType.WORKING` | Short-term context for the current session. |
| `skill` | `MemoryType.SKILL` | A synthesized procedure: steps, rules, and an optional tool schema. |

Passing `memory_type="skill"` to `remember()` marks a normal text atom as a skill. `remember_skill()` is the structured writer: it fills the text, metadata, triples, and type together, and it can pin a stable id.

Unfiltered `query()` still searches skills along with every other type. Pass `memory_type="skill"` when the caller wants procedures only.

---

## Writing a skill

```python
from epochdb import EpochDB

with EpochDB(storage_dir="./memory") as db:
    skill_id = db.remember_skill(
        skill_name="refund_order",
        description="Refund a paid order after checking it has not already been refunded.",
        steps=[
            {
                "step_num": 1,
                "action": "Look up the order",
                "details": "Confirm it is paid and has no existing refund.",
            },
            {
                "step_num": 2,
                "action": "Call the payments refund tool",
                "details": "Pass the order id and the captured amount.",
            },
        ],
        decision_rules=[
            "Never refund more than the captured amount.",
        ],
        tool_schema={
            "name": "refund_order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                },
                "required": ["order_id"],
            },
        },
        skill_id="skill-refund-order",
    )
```

`remember_skill()` returns the atom id. When `skill_id` is set, that string is the atom id, so a later write of the same skill replaces the same record.

`AsyncEpochDB.remember_skill()` takes the same arguments and runs the sync method in a worker thread.

### Arguments

| Argument | Role |
| --- | --- |
| `skill_name` | Canonical name. Also used as the graph subject when no custom triples are passed. |
| `description` | What the procedure is for. Stored in metadata and in the text payload. |
| `steps` | Ordered list of dicts. Each step may include `step_num`, `action`, and `details`. |
| `decision_rules` | Short constraints the agent should apply while running the skill. |
| `tool_schema` | JSON-schema style description of the tool this skill drives. Defaults to `{}`. |
| `metadata` | Extra fields merged in first. Reserved keys below are then overwritten. |
| `skill_id` | Stable atom id. Also stored as `metadata["skill_id"]` and `metadata["process_id"]`. |
| `triples` | Optional graph triples. When omitted, EpochDB builds the default skill triples. |

### Text payload

The atom text is a single searchable string:

```text
SKILL [refund_order]: Refund a paid order after checking it has not already been refunded.
Steps: Step 1: Look up the order — Confirm it is paid and has no existing refund.; Step 2: Call the payments refund tool — Pass the order id and the captured amount.
Rules: Never refund more than the captured amount.
```

`step_num` falls back to the list position (starting at 1) when a step omits it. The rules line is included only when `decision_rules` is non-empty. This string is what the embedding model encodes, so a later semantic query can match the skill by name, description, or step wording.

### Metadata

`remember_skill()` writes these fields onto the atom:

| Key | Value |
| --- | --- |
| `skill_name` | The name you passed. |
| `title` | Existing `metadata["title"]`, otherwise `skill_name`. |
| `description` | The description you passed. |
| `steps` | The steps list, unchanged. |
| `decision_rules` | The rules list, or an existing `metadata["decision_rules"]`. |
| `tool_schema` | The schema you passed, or an existing `metadata["tool_schema"]`, or `{}`. |
| `type` | Always `"skill"`. |
| `skill_id` / `process_id` | Set only when `skill_id` is passed. |
| `created_at` | Unix timestamp, preserved if the caller already set one. |

`metadata["type"]` is the legacy marker. Older archives used `"process_summary"` for the same role. On cold-tier load, both `"skill"` and `"process_summary"` restore as `MemoryType.SKILL`.

### Knowledge-graph triples

When `triples` is omitted, EpochDB records:

- `(skill_name, "is_skill", skill_id or skill_name)`
- `(skill_id or skill_name, "has_step_{n}", action)` for each step that has an action

Quotes are stripped from the action text before it becomes a triple object. Pass `triples=` yourself when the skill should link to domain entities (`refund_order` → `applies_to` → `Order`) instead of, or in addition to, these structural edges.

---

## Reading skills back

### `get_skill(skill_name)`

Lookup is exact and case-insensitive via the hot-tier `SkillIndex`, then cold-tier skill refs, matching any of:

1. `metadata["skill_name"]`
2. `metadata["title"]`
3. `metadata["skill_id"]`
4. `metadata["process_id"]`
5. the atom id

If none of those match, `get_skill` runs a semantic `query` limited to `memory_type="skill"` and `filters={"skill_name": skill_name}`, and returns the first hit. An empty name returns `None`.

### `list_skills()`

Uses the hot-tier `SkillIndex` plus a cold-tier Parquet scan for skill-typed atoms (`memory_type` / `metadata["type"]` / `skill_name`). It does **not** walk the full timeline.

Deleted atoms (`metadata["_deleted"]`) are skipped. Results are unique by atom id and sorted newest first.

### Prompt injection

`get_hot_summary_snapshot(user_id=None)` builds a short block for a system prompt. With a `user_id`, it includes up to five profile facts from `get_user_profile`. It then appends up to five skills from `list_skills()`:

```text
## Synthesized Agent Skills (EpochDB)
- refund_order: SKILL [refund_order]: Refund a paid order...
```

Each skill line is the name plus the first 120 characters of the atom text.

---

## Durability

`remember()` assigns `memory_type` after the atom is created, then appends the atom to the write-ahead log again so a restart keeps the type. The cold-tier Parquet files store `memory_type` in its own column.

Older Parquet files omit that column. Reload then uses `metadata["type"]`, and falls back to `"general"` when that is missing too. A stored value of `"process_summary"` is read back as `"skill"`. An unrecognized type string becomes `MemoryType.GENERAL`.

---

## Filtering at query time

```python
procedures = db.query(
    "how do I refund an order",
    k=5,
    memory_type="skill",
)
```

The same `memory_type` argument exists on `AsyncEpochDB.query`, the LangChain tools in `epochdb.core.tools`, and the remote client. An unknown type string is ignored and the query runs unfiltered.

`remember(..., memory_type="skill")` is enough when the caller already has a flat text blob. Use `remember_skill()` when the procedure has steps, rules, and a tool schema that later code will read from `metadata`.

---

## Agent tooling (MCP & LangChain)

Skills and profiles are exposed as first-class tools:

| Surface | Tools |
| --- | --- |
| MCP | `epochdb_remember_skill`, `epochdb_get_skill`, `epochdb_list_skills`, `epochdb_remember_user_profile`, `epochdb_get_user_profile`, `epochdb_get_hot_summary_snapshot` |
| LangChain (`get_epochdb_tools`) | Same names as StructuredTools |

Pass `memory_type="skill"` on `epochdb_remember` / `epochdb_query` when you want typed filters without the structured skill writer.
