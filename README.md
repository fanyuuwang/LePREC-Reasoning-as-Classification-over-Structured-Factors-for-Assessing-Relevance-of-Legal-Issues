# LePREC

Official implementation of [*LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues*](https://aclanthology.org/2026.acl-long.350/), ACL 2026.

LePREC is a neuro-symbolic framework for legal issue relevance prediction.  It converts legal reasoning factors into natural-language questions, obtains a score for each question, and uses a transparent linear classifier to predict whether a legal issue is relevant to a case.

> **Data access:** the framework code is publicly released under the MIT License. The LIC dataset, including case text and expert annotations, is **available upon request** and is not included in this repository. Please kindly send your:
> - Your name, institution, and institutional email address.
> - The research purpose and intended use of the data.
> - A statement that you will not redistribute the data or use it outside the approved purpose.

> To request access, please kindly contact Fanyu Wang (fanyu.wang@monash.edu) or Dr. Lizhen Qu (lizhen.qu@monash.edu)
> The data are not covered by the MIT License.

## Repository layout

```text
src/leprec/
  prompts.py        # reasoning-question generation and parsing
  features.py       # question-score feature construction
  classifier.py     # sparse linear relevance classifier
  incremental.py    # incremental reasoning workflow
  evaluation.py     # cross-validation evaluation
  validation.py     # checks for approved LIC data releases
  cli.py            # command-line interface
LICENSE             # MIT License for the code
```

## Installation

```bash
git clone <your-repository-url>
cd leprec
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The package exposes the `leprec` command-line interface. Use `leprec --help` to view the available commands.

## Dataset

LIC contains legal case–issue examples used in the paper, together with the supporting case text, expert annotations, reasoning questions, and question scores needed for release validation and evaluation. It is not distributed in the public code repository.

Researchers seeking access should follow the process in [DATA_ACCESS.md](DATA_ACCESS.md). Approved recipients should keep the dataset outside the public repository and comply with the agreed data-use conditions.

After access is approved, arrange the supplied files as follows:

```text
/path/to/LIC/
  LIC.json
  LICU.json
  reasoning_questions.json
```

Validate an approved data release:

```bash
leprec validate-data --release-root /path/to/LIC
```

Run the framework evaluation:

```bash
leprec evaluate-release \
  --release-root /path/to/LIC \
  --output results/leprec_cv_metrics.json
```

## Framework workflow

1. Generate or provide structured legal-reasoning questions for each candidate issue.
2. Score the questions against the facts of the case.
3. Construct a question-score feature vector for every case–issue pair.
4. Fit and evaluate the linear relevance classifier.

The supplied commands validate the correspondence between the case–issue labels, reasoning questions, and score matrix before evaluation.

## Citation

If you use this code or the LIC dataset, please cite:

```bibtex
@inproceedings{wang-etal-2026-leprec,
  title = {Le{PREC}: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues},
  author = {Wang, Fanyu and Kang, Xiaoxi and Burgess, Paul and Srivastava, Aashish and Arora, Chetan and Trakic, Adnan and Soon, Lay-Ki and Hossain, Md Khalid and Qu, Lizhen},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year = {2026},
  pages = {7701--7736},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2026.acl-long.350},
  url = {https://aclanthology.org/2026.acl-long.350/}
}
```

## License

The source code is released under the [MIT License](LICENSE). The LIC dataset is not covered by that license; it is available only under the data-access process described in [DATA_ACCESS.md](DATA_ACCESS.md).
