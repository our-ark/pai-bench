# Security Policy

Report suspected vulnerabilities through GitHub's private security advisory
flow. Do not open a public issue for a credential leak, arbitrary command
execution, target-oracle disclosure, or evaluator isolation failure.

Target and evaluator adapters execute local commands configured by the user.
They are integration boundaries, not sandboxes. Review adapter code and run it
with only the credentials and filesystem access required for the experiment.
