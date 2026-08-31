# PAI-Bench vNext development suite

This directory is generated and intentionally **not frozen**. It isolates the
construct-validity changes proposed after PAI-Bench v1.0: atomic identity
baselines, a balanced identity composition-depth ladder with matched neutral
controls, assisted/unassisted resistance, credential-based governance, and
semantic-equivalent decision prompts.

The current development matrix contains 8 identities in 4 matched pairs and
25 focused probes per identity. Composition component order is rotated across
the four pairs while remaining matched within each pair.

Do not compare these results directly with the frozen v1.0 headline score.
Use the suite for development pilots only until a new protocol is declared and
frozen. Regenerate it with:

```bash
identity-benchmark generate-vnext development/vnext
```
