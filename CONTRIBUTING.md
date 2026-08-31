# Contributing

PAI-Bench changes should preserve provider neutrality, target-oracle
separation, deterministic release generation, and reproducible evaluation.

## Local checks

Use Python 3.11 or newer and run:

```bash
python -m unittest discover -s tests
bin/release --check
```

Changes to frozen identities, probes, bindings, splits, or scoring rules must
update the generator provenance and release protocol. Do not tune against
responses from a frozen split.

Do not commit credentials, private identities, conversation logs, machine-local
paths, model outputs containing private data, or untracked experiment state.
Use synthetic fixtures for tests and examples.
