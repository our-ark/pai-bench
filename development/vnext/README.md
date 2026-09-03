# PAI-Bench vNext development suite

This directory is generated and intentionally **not frozen**. It isolates the
construct-validity changes proposed after PAI-Bench v1.0: atomic identity
baselines, a balanced identity composition-depth ladder with matched neutral
controls, assisted/unassisted resistance, credential-based governance, and
semantic-equivalent decision prompts.

The current development matrix contains 8 identities in 4 matched pairs and
25 focused probes per identity. Composition component order is rotated across
the four pairs while remaining matched within each pair. Neutral project facts
vary across pairs, are installed once through `set_startup_context()`, and are
reloaded from isolated target state at each fresh session. They are not repeated
inside the neutral probe message. This matches persistent delivery and recency
more closely while keeping neutral facts separate from personal identity.
Reports keep identity and neutral joint compliance separate at every depth and
provide the descriptive neutral-minus-identity gap; controls remain outside the
headline identity score.

Do not compare these results directly with the frozen v1.0 headline score.
Use the suite for development pilots only until a new protocol is declared and
frozen. Regenerate it with:

```bash
identity-benchmark generate-vnext development/vnext
```
