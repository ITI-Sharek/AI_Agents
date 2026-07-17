# Root Cause Analysis: LLM Returns 14 General Skills Instead of 5

## 1. Current Configuration

**Prompt** (`prompts.py`): Contains clear language — "Output exactly 5 entries, one per category above" (line 70), "never add a 6th" (line 71), "You must return exactly 5 GeneralSkill objects" (line 160). However, the prompt also includes a long section on dynamic `framework_skills` (lines 178–251) that explicitly says "output one skill entry for EVERY distinct framework/library." This creates a conceptual parallel: the LLM sees both a "fixed 5" pattern and a "dynamic, one-per-item" pattern, both expressed as arrays.

**Output Schema** (`schemas.py`): `SkillProfilingResult` defines:

```python
general_skills: list[GeneralSkill]
```

where `GeneralSkill.name` is a **Literal of exactly 5 values**. The list itself has **no length constraint** — no `min_length`, no `max_length`, no uniqueness constraint on names.

**Structured output** (`graph.py:44`):

```python
structured = get_llm().with_structured_output(SkillProfilingResult)
```

LangChain's `with_structured_output` converts the Pydantic schema to a **JSON Schema** and passes it as a tool/function definition to the LLM. The JSON Schema for `general_skills` becomes:

```json
{
  "type": "array",
  "items": { "$ref": "#/$defs/GeneralSkill" }
}
```

No `minItems`, no `maxItems`, no `uniqueItems`.

**Validator** (`router.py:22-36`): An application-layer function `_validate_profiling_result` checks `len != 5` and name order **after** the LLM output is already parsed and accepted by Pydantic.

---

## 2. Why the Model Can Return 14 Entries

The model can return 14 entries because **nothing in the programmatic contract prevents it**. The chain of enforcement is:

| Layer | Enforces | Enforces exactly 5? |
|-------|----------|---------------------|
| Prompt text | Advisory | No (soft suggestion) |
| JSON Schema (tool def) | Type correctness | **No** (no length bound) |
| Pydantic parsing | Type correctness | **No** (accepts any length) |
| Application validator | Length + order | Yes, but **too late** |

The JSON Schema sent to the LLM says "produce an array of objects that look like `GeneralSkill`." The LLM, trained to produce schema-conforming output, treats cardinality as a free parameter. If the evidence is rich or multi-repo, the LLM may feel it needs more entries to cover the detail — and the schema permits this.

Critically, the `Literal` on `name` constrains **what values each entry's name can take**, but not **how many entries** nor **whether names are unique**. The model could produce 14 entries all with name `"Clean Code"`, and it would still pass Pydantic validation.

---

## 3. Where the Problem Originates (Ranked)

1. **The Pydantic schema** — most direct cause. `list[GeneralSkill]` imposes zero cardinality or uniqueness constraints.
2. **The structured output implementation** — translates the schema to JSON Schema without adding `minItems`/`maxItems`/`uniqueItems`, so the LLM receives an unbounded contract.
3. **The prompt** — contributes marginally. While it says "exactly 5", it is overshadowed by the schema in the LLM's decision hierarchy (tool calling drives output, not prompt text).
4. **The LLM behavior** — follows the schema's structural signals. Given an unbounded array schema, it is compliant to produce any number of entries.
5. **Combination**: The schema is the primary cause; the implementation and prompt are secondary contributors. The schema's unbounded list is the single necessary condition — without it, the problem cannot occur.

---

## 4. How `list[GeneralSkill]` Influences Behavior

`list[GeneralSkill]` tells both the LLM and Pydantic: "any number of `GeneralSkill` objects is valid." Specifically:

- **JSON Schema generated**: `"type": "array", "items": { ... }` with **no `minItems`** — the default is 0.
- **Pydantic behavior**: parses any non-negative length. No `ValidationError` for 14 entries.
- **LLM interpretation**: the schema defines a **validity envelope**; cardinality is unrestricted within it. The LLM will generate as many entries as it feels the evidence warrants, because the schema doesn't signal that cardinality is a fixed constraint.

This is in contrast to, say, a `tuple[GeneralSkill, GeneralSkill, GeneralSkill, GeneralSkill, GeneralSkill]` type, which would fix the cardinality at 5 in the JSON Schema as `"prefixItems"` with exactly 5 positions.

---

## 5. Why the Validator Catches Only the Symptom

The validator `_validate_profiling_result` runs **after** the LLM output is parsed, accepted by Pydantic, and returned from `_invoke_llm`. It detects the wrong cardinality but:

- Tokens and latency have already been spent.
- The LLM is not asked to retry (the retry logic in `_invoke_llm` only catches `ValidationError`/`ValueError`/`TypeError`, and no exception is raised for 14 entries).
- The router returns a `failed` response (`"invalid_profiling_output"`) — the flow stops with an error rather than self-correcting.

The validator is a **guard rail**, not a **constraint**. It catches the symptom (wrong count) but does not prevent the root cause (unbounded schema). It would catch any count ≠ 5, but by then the damage is done.

---

## 6. Prompt Ambiguity That Encourages Multiplicity

The prompt contains two adjacent structural patterns:

```
"Output exactly 5 GeneralSkill objects" (line 160)
"output one skill entry for EVERY distinct framework/library" (line 180)
```

Both produce list-type outputs. The LLM may **generalize the "one per item" pattern** from framework_skills to general_skills, especially when:

- Evidence comes from multiple repos (line 147 says "describes ONE contributor across ALL repos" but the schema doesn't enforce uniqueness).
- The prompt uses language like "Cross-repo aggregation" which may suggest the LLM should synthesize across repos — and an LLM can misinterpret this as needing multiple entries per category (one per repo) before aggregating, or may fail to aggregate completely.
- The HARD RULES say "one per category" but the schema allows more. The schema signal overrides the text signal for many LLMs.

---

## 7. The Output Format Gives Too Much Freedom

The output format is an **array with no fixed cardinality**. When an LLM uses tool calling / structured output, the JSON Schema is the **primary signal** — it defines the space of valid outputs. Prompt text is secondary. The schema tells the LLM:

> "You may produce any number of `GeneralSkill` objects, each with a name from this set of 5."

This is effectively a **permission to produce 1, 5, 14, or 100 entries**. The LLM will exercise that freedom based on its own heuristics (e.g., "more repos = more entries needed, perhaps each repo gets its own general skills entry").

---

## 8. Architectural Weakness of a Free-Form List for Fixed Skills

The architectural mismatch is:

| Property | Actual requirement | Schema representation |
|----------|-------------------|----------------------|
| Cardinality | Exactly 5 | Unbounded |
| Identity | Each skill is uniquely identified by name | No uniqueness constraint |
| Order | Must match a fixed sequence | No order enforcement |
| Mutability | Set is fixed and known at compile time | Schema treats it as dynamic |

Using `list[GeneralSkill]` for a set that is **fixed, known, and enumerated in a Literal** is a type system mismatch. The Literal is already half the solution (it constrains the name values), but it is paired with a container that has none of the necessary structural guarantees. The list type is designed for variable-length homogeneous data; the five general skills are **fixed-length, versioned, named slots**.

This is the equivalent of modeling `{"a": int, "b": int}` as `list[int]` and hoping the LLM always produces exactly two elements in the right order.

---

## 9. List vs Fixed Named Fields: Trade-offs

| Aspect | `list[GeneralSkill]` | Five named fields |
|--------|----------------------|-------------------|
| Cardinality enforcement | None (needs `min_length`/`max_length`) | **Built-in** — exactly 5 fields |
| Uniqueness enforcement | None (needs `@field_validator`) | **Built-in** — each field is distinct |
| Order enforcement | None (needs post-check) | **Built-in** — fields are named |
| Iterability | **Natural** — `for gs in result.general_skills` | Needs `model_dump()` or a helper |
| JSON Schema generated | Unbounded array | Fixed object with 5 properties |
| LLM friendliness | LLMs generate arrays easily | LLMs generate named fields easily too |
| Schema verbosity | Compact | Verbose (5 near-identical fields) |
| Extensibility (if set changes) | Change Literal only | Add/remove field + Literal change |
| Self-documenting | Array element is opaque | Each field name tells the story |

The list trades **structural guarantees** for **iteration convenience**. This trade-off is reasonable only when the cardinality is genuinely variable — which it is not for general skills.

---

## 10. Actual Root Causes (Ranked by Likelihood)

**#1 — Schema: `list[GeneralSkill]` has no cardinality constraint (PRIMARY)**
The Pydantic model uses `list[GeneralSkill]` without `min_length=5, max_length=5`. This translates to an unbounded JSON Schema array. The LLM is permitted (by the schema) to produce any number of entries. This is the **single sufficient cause** — if the schema enforced fixed length, the problem could not occur regardless of the prompt or LLM behavior.

**#2 — Structured output implementation does not enrich the schema with bounds (SECONDARY)**
`with_structured_output` converts the Pydantic schema naively to JSON Schema. It could add `minItems`/`maxItems` for lists with `min_length`/`max_length` in the Pydantic `Field`, but since those are absent, the generated schema is unbounded. The implementation does not augment the schema to encode the prompt's cardinality requirements.

**#3 — The prompt's "exactly 5" is text-level, not schema-level (TERTIARY)**
When tool calling is active, the JSON Schema is the binding contract. The prompt text is advisory. An LLM will prioritize schema conformance over prompt compliance. The prompt says "exactly 5" but the schema says "any number" — the schema wins.

**#4 — The Literal on name constrains values but not multiplicity (CONTRIBUTING)**
The Literal correctly restricts each entry's name to one of five values, but it does not prevent duplicate entries with the same name. The LLM can produce `"Clean Code"` three times and still pass schema validation. This is a false sense of security — developers see "Literal of 5 values" and assume it enforces the 5-skills constraint, but it only constrains each individual entry, not the set as a whole.

**#5 — The retry mechanism catches the wrong exception class (CONTRIBUTING)**
`_invoke_llm` catches `ValidationError` to trigger a retry, but no `ValidationError` is raised for 14 entries (Pydantic accepts it). The actual failure is caught by the application-layer validator afterward, which returns a terminal error instead of allowing a retry. The validation logic is **correct but misplaced** — it should be part of the schema or at least trigger a retry, not just fail.

**#6 — The parallel with `framework_skills` (MINOR)**
The prompt's section on `framework_skills` uses the same array pattern but with dynamic cardinality ("one per framework"). This may encourage the LLM to treat `general_skills` similarly — as a dynamic array whose length depends on input characteristics — even though the prompt explicitly says "exactly 5."

---

### Conclusion

The root cause is a **schema design failure**: representing a fixed, known-in-advance set of 5 items as an **unbounded array** (`list[GeneralSkill]`). The LLM follows what the schema permits, and the schema permits any number of entries. Prompt instructions are advisory; schema constraints are binding. Every other factor (retry misalignment, prompt ambiguity, missing uniqueness enforcement) is downstream or secondary.
