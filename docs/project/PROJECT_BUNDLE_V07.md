# Project evidence contract 0.7

The `0.7-project` schema is an envelope around an unchanged `0.6-algorithm` transaction experiment and an extractive financial retrieval result. Software distribution version and evidence-schema version are separate fields; 0.7.2rc1 also continues to read legacy schemas.

## Files and invariants

`risk/` contains exactly the required 0.6 artifacts and its original bundle manifest. The root contains run_metadata, lineage, corpus, questions, retrieval_evidence, release_gate, review_report and a bundle manifest. The parent manifest covers every nested artifact and the child manifest. It does not hash itself. Unknown files, symlinks, missing files and altered content are rejected.

The parent run ID identifies the review assembly. `lineage.json` binds the original risk run ID and manifest hash, the financial corpus file hash and query file hash. The child run metadata retains the actual training code identity and environment. The parent records the assembly code identity; assembly does not represent a new training experiment.

`project-run` trains the risk child and assembles retrieval in one command. `project-assemble` reuses a verified child. Both publish the final directory only after component and joint verification. Failed attempts leave no completed project directory. Completed IDs are not overwritten; a per-ID lock prevents competing finalization. Stale lock recovery after an abrupt process kill is a manual check, not an automatic deletion.

## Verification boundary

The verifier checks exact inventories, hashes, child semantic validation, rebuilt retrieval citations and scores, exact archived questions, lineage, version relations, fixed execution wording, code identity format, gate consistency and regenerated HTML. It optionally verifies the external local review journal. These checks cannot independently authenticate a producer who controls and rewrites all evidence and hashes. They do not execute joblib files or rerun model training.

The combined gate inherits the child research/production restrictions and adds `financial_analysis.status=extractive_only`. Supported passages and unsupported questions are both valid outcomes. A financial extract is not linked to a specific anonymous transaction. A blocked production gate is a policy restriction, not a failed installation.

The local review page displays the transaction classification and each financial document's classification. It presents both sources, model comparisons and links to raw fault/recovery evidence. Reviews bind the parent manifest. Downloaded ZIPs include only declared artifacts; review journals stay beside the immutable bundle.

## Remaining scope

This closes the missing common runtime/evidence envelope. It does not supply a generative LLM, a general financial-PDF parser, licensed institution data, real transaction entity relationships, or external model approval. None of those is inferred from joint verification success.
