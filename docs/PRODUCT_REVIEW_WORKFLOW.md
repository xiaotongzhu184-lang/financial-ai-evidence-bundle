> Current 0.7 entry point: `project-smoke` / `project-run` / `project-assemble`. See [README](../README.md) and [the project contract](project/PROJECT_BUNDLE_V07.md). The historical examples below retain their original schema scope.

# Reviewer workflow and acceptance gates — 0.4.0rc1

The intended user is a researcher or model-risk reviewer deciding whether an experimental result has sufficient evidence for a **stated research use**. Local evidence acceptance does not authorize production use.

## Operator and reviewer journey

1. Acquire the separately licensed ULB historical CSV with `scripts/fetch_ulb.py`, or use the generated fixture through `research-smoke`. The real-data classification is bound to the known publisher URL, exact SHA-256 and canonical source profile. Other institutional schemas need a separate adapter; relabelling a synthetic CSV is rejected.
2. Run `research-run`. DuckDB checks source structure and basic validity. The evaluator uses separate chronological training, calibration, validation and test partitions, fitted preprocessing, three fixed model configurations, validation-based selection and operating thresholds. All held-out model results survive selection.
3. Keep every fixed-model stress result and measure bounded repeated-feature inference load. The load is not new accuracy evidence. Review alert counts and missed cases; no handling-time or monetary saving is estimated without actual inputs.
4. Retrieve literal excerpts from the archived multi-issuer SEC Company Facts-derived corpus. The interface cites the source and abstains on unsupported query terms. There is no LLM answer generator or validated entity linkage between SEC facts and ULB transactions.
5. Run `verify`. Strict file lists, hashes, partition identities, calibration maps, metric arithmetic, three required stress scenarios, retrieval outputs and the policy fields are checked. Incoming joblib files are hash-checked and never unpickled by verification or the viewer.
6. Open the loopback-only page with `serve`. Inspect metrics, false alerts, missed cases, stress results, capacity and citations. Record accept/request-changes/reject plus a reason. The local record links to the bundle manifest and previous record hash; it is self-declared, not authenticated proof of reviewer identity. Export rechecks the evidence.

## Separate outcomes

| Outcome | Meaning | Does not imply |
| --- | --- | --- |
| Evidence completeness: pass | Required artifacts and supported calculations are internally consistent | Independent source authentication or valid business performance |
| Research suitability: historical_benchmark_only | Exact declared historical ULB source is bound to the research evidence | Current institutional representativeness |
| Research suitability: smoke_only | Generated rows can check software behavior | Real fraud performance |
| Production approval: blocked | No institutional pilot and no operational authorization | A software or CI failure |

A stress failure is a retained negative finding, not permission to delete the experiment. The local review choice cannot change the production status.

## Acceptance evidence

The release candidate is checked by automated source/wheel tests, clean-environment execution, actual HTTP review/export tests, semantic tamper regressions and three fixed-seed ULB runs. Exact commands and their raw outputs belong in the versioned delivery archive; do not infer current test success just from this descriptive document. Independent acceptance should include downloading the same source and reproducing its core statistics.

## Retained legacy path

`contract-run` continues the 0.3 generated transaction-network, decision-impact and optional GraphSAGE experiment. Its 0.3 contract and model/data cards describe that separate path. It must not be presented as if the ULB table contained customer or merchant network identifiers.
