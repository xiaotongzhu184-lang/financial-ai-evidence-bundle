# Security policy

## Supported version

Security fixes are evaluated against the current `0.7.2rc1` release candidate. Older research schemas remain readable for reproducibility but are not maintained as independent supported releases.

## Reporting a vulnerability

Please use the repository's private vulnerability-reporting form:

https://github.com/xiaotongzhu184-lang/financial-ai-evidence-bundle/security/advisories/new

Do not post credentials, exploit details, personal data or unreleased evidence bundles in a public issue. A non-sensitive defect without security impact can be reported through GitHub Issues.

Include the affected version, operating system, Python version, entry point, minimal reproduction and expected impact. Reports are assessed against the local-only research deployment described in the README; the repository does not claim an internet-facing service or production fraud system.

## Trust boundaries

- The review server binds to loopback and rejects foreign Host and Origin values.
- Evidence verification checks hashes, schemas and project-specific semantic relationships. It is not a digital signature or source-authentication service.
- Model files are serialized Python objects and can execute code when loaded. Only `score-batch --trust-models --same-feature-space` loads them, after the operator makes an explicit trust decision.
- Public fixtures contain generated data and small SEC-derived fact records. The ULB/Worldline raw CSV is downloaded separately and is not part of this repository.
