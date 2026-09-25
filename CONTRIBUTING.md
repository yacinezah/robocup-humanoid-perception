# Contributing and reproducing

Use a separate branch and state the scientific or engineering question before
changing a model, evaluator or protocol. Never silently replace frozen tables,
thresholds, checkpoints or report conclusions with a newer result.

Run `python -m pytest` and `python scripts/audit_publication.py`. New inference
paths must document shape, colour order, normalization, coordinates, class order
and missing-input behavior. Include synthetic tests that require no private data.

Training and robot integration require separate explicit authorization and
eligible data. None of the examples contacts a robot or reserves a GPU. Do not
submit datasets, protected manifests, credentials, host addresses, optimizer
states or private logs. Model assets belong in reviewed Releases, not Git.

Research changes and the isolated C++ production-review patch must remain
separate. A passing smoke test is not an accuracy evaluation or integration approval.

Preserve attribution and component licences. Report errors by identifying the
public report/commit, evidence class and affected quantity. Publish corrections
as a new version, keeping earlier citation targets available.
