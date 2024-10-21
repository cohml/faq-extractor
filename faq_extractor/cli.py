import argparse
from pathlib import Path

from .utils import (
    CONFIG_JSON_REQUIRED_FIELDS,
    validate_column_names,
    validate_config,
)


parser = argparse.ArgumentParser(
    description=(
        "Extract the top n most frequent FAQs from a dataset using a "
        "pipeline of language models. Data must be in CSV format. "
        "Requires GPT access."
    ),
)

parser.add_argument(
    "--questions-csv-path",
    default="hf://datasets/bitext/Bitext-wealth-management-llm-chatbot-training-dataset/bitext-wealth-management-llm-chatbot-training-dataset.csv",
    help=(
        "Path to CSV file enumerating questions from which to "
        "extract the top n most common FAQs. Default: %(default)s"
    ),
)

parser.add_argument(
    "--questions-column-name",
    default="instruction",
    help=(
        "Name of column in questions CSV file containing question "
        "texts. Default: %(default)s"
    ),
)

parser.add_argument(
    "--answers-column-name",
    help=(
        "Name of column in questions CSV file containing answer "
        "texts. Default: %(default)s"
    ),
)

parser.add_argument(
    "--topics-column-name",
    help=(
        "Name of column in questions CSV file containing "
        "topic/intent labels for each question. If such a column "
        "exists, it will help in setting the optimal number of "
        "clusters. Default: %(default)s"
    ),
)

parser.add_argument(
    "--output-json-path",
    default=Path.cwd() / "faqs.json",
    type=Path,
    help=(
        "Path to JSON file with extracted FAQs and other related "
        "data. Default: %(default)s"
    ),
)

parser.add_argument(
    "--config-json-path",
    type=validate_config,
    help=(
        "Path to JSON file enumerating metadata needed to "
        "interact with OpenAI's GPT. Must contain the following "
        f"fields: {CONFIG_JSON_REQUIRED_FIELDS}"
    ),
    required=True,
)

parser.add_argument(
    "--n-faqs",
    default=5,
    type=int,
    help=(
        "Number of most-frequent FAQs to extract. Default: "
        "%(default)s"
    ),
)

parser.add_argument(
    "--n-example-questions",
    default=5,
    type=int,
    help=(
        "Number of questions from the source data CSV file to include "
        "as examples for each extracted FAQ. Default: %(default)s"
    ),
)

parser.add_argument(
    "--embedding-model-name",
    default="paraphrase-MiniLM-L6-v2",
    help=(
        "Name of HuggingFace model to use when embedding "
        "questions. Must be compatible with "
        "``sentence-transformers``. Default: %(default)s"
    ),
)

parser.add_argument(
    "--spelling-correction-model-name",
    default="oliverguhr/spelling-correction-english-base",
    help=(
        "Name of HuggingFace model to use for optional spelling "
        "before computing evaluation metrics. Default: %(default)s"
    ),
)

parser.add_argument(
    "--n-principal-components",
    default=50,
    type=int,
    help=(
        "Number of dimensions to project semantic vectors onto "
        "before clustering. Default: %(default)s"
    ),
)

parser.add_argument(
    "--n-clusters",
    type=int,
    help=(
        "Number of clusters to use when identifying potential "
        "FAQs. This parameter interacts with ``--n-faqs``, in that "
        'the closer this value aligns with the "true" number of '
        "FAQs in the dataset, the better the algorithm should work. "
        "If unspecified, we will try to infer the optimal value from "
        "data file, else we will set arbitrarily to 25."
    ),
)

parser.add_argument(
    "--n-max-clusters",
    default=25,
    type=int,
    help="Maximum number of clusters to create during clustering.",
)

parser.add_argument(
    "--evaluation-metric",
    choices=["bertscore", "rouge"],
    help=(
        "Metric used to evaluate how well each extracted FAQ "
        "captures the semantics of a group of related questions. "
        "If unspecified, no evaluation will be performed. Default: "
        "%(default)s"
    ),
)

parser.add_argument(
    "--do-spelling-correction",
    action="store_true",
    help=(
        "Whether or not to correct the spelling of questions "
        "before computing evaluation metrics. If your data "
        "contains typos or unconventional spellings, failure to "
        "correct for this may artificially lower the metrics. "
        "But performing spelling correction will increase the "
        "program's runtime. Default: %(default)s"
    ),
)

parser.add_argument(
    "--use-azure-openai",
    action="store_true",
    help= "Whether your GPT deployment is accessible through Azure OpenAI."
)


parser.add_argument(
    "--seed",
    default=42,
    type=int,
    help="Seed value for reproducibility. Default: %(default)s",
)

args = parser.parse_args()

validate_column_names(
    args.questions_column_name,
    args.answers_column_name,
    args.topics_column_name,
)
