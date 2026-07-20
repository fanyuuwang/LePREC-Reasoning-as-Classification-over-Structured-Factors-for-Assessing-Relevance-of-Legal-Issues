# LEPREC Repository

This directory contains a clean implementation of the framework in *LEPREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues* (Wang et al., ACL 2026).

LEPREC has four stages:

1. Construct fact prefixes for incremental candidate-issue generation.
2. Build and parse the constrained `Whether ...` issue-generation prompts.
3. Treat Yes/No verifier probabilities for a shared reasoning-question pool as structured features.
4. Fit and inspect correlation-aware standard linear classifiers using nested stratified five-fold cross-validation.

The package deliberately excludes credentials, cloud authentication, model checkpoints, caches, generated logs, and unrelated experimental scripts.

## Install

From this directory, create an isolated environment and install the package:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install "numpy>=1.24" "scikit-learn>=1.3"
.venv/bin/python -m pip install -e .
```

## Prepare the dataset release

The script reads the evaluation inputs from the original research workspace and writes only into the sibling `release_dataset/` directory:

```bash
.venv/bin/python -m leprec.cli prepare-data \
  --source-root ../dataset/LegalSemi \
  --release-root ../release_dataset
```

The resulting manifest verifies the expected 1,188 LICL labels, 2,464 reasoning questions, their Phi-4 feature matrix, and the source-derived LICU export.

The primary human-readable datasets are `../release_dataset/LIC.json`, `../release_dataset/LICU.json`, and `../release_dataset/reasoning_questions.json`. LIC records contain `case_id`, `case_name`, point-form `facts`, and issue objects with both `relevance_label` (`Relevant`/`Irrelevant`) and the original binary `label`. LICU records use the same case fields and identify each issue as `Source-extracted (unlabeled)`; no expert binary label is invented. The reasoning-question file contains the ordered `id`, `question`, and `explanation` objects used to form the feature columns.

## Validate the complete public dataset

Run this before evaluation or publication:

```bash
.venv/bin/python -m leprec.cli validate-data \
  --release-root ../release_dataset
```

The validator checks the complete data contract: LIC labels, the exact `0/1` to `-1/1` feature-label encoding and order, reasoning-question metadata/order, probability-score bounds, LICU's unlabelled status, and the release manifest hashes.

## Reproduce the linear evaluation

```bash
.venv/bin/python -m leprec.cli evaluate-release \
  --release-root ../release_dataset \
  --output ../release_dataset/expert_annotations/leprec_cv_metrics.json
```

`evaluate-release` validates the release before fitting the nested cross-validation models. The Phi-4 score matrix is a released derived artifact. Regenerating it with another verifier model is optional: use `leprec.incremental.fact_prefixes`, `leprec.prompts.build_incremental_prompt`, and a model-specific scorer to produce one probability score per reasoning question and issue-fact pair. For a custom already-validated feature file, use the lower-level `evaluate --features ...` command.

## Citation

```bibtex
@inproceedings{wang-etal-2026-leprec,
  title = {LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues},
  author = {Wang, Fanyu and Kang, Xiaoxi and Burgess, Paul and Srivastava, Aashish and Arora, Chetan and Trakic, Adnan and Soon, Lay-Ki and Hossain, Md Khalid and Qu, Lizhen},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2026},
  pages = {7701--7736},
  doi = {10.18653/v1/2026.acl-long.350}
}
```

No software licence has been selected yet. Add an author-approved `LICENSE` file before making this repository public.
