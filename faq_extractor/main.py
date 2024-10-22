import json
from typing import Optional

import pandas as pd

from .cli import args
from .models import (
    embedding_model,
    kmeans,
    pca,
    spelling_correction_model,
    summarization_model,
)
from .pipeline import TopNFAQExtractionPipeline
from .utils import logger


def load_questions_csv(
    questions_csv_path: str,
    questions_column_name: str,
    answers_column_name: Optional[str],
    topics_column_name: Optional[str],
) -> pd.DataFrame:
    """
    Load questions, answers, and topics from a CSV file.

    Parameters
    ----------
    questions_csv_path : str
        Path to the CSV file containing questions.
    questions_column_name : str
        Name of the column in the CSV file that contains question texts.
    answers_column_name : Optional[str]
        Name of the column containing answers, if applicable.
    topics_column_name : Optional[str]
        Name of the column containing topic labels, if applicable.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns "question", "answer", and "topic" (if present).

    Raises
    ------
    ValueError if the required columns are not present.
    """
    columns = {questions_column_name: "question"}
    if answers_column_name is not None:
        columns[answers_column_name] = "answer"
    if topics_column_name is not None:
        columns[topics_column_name] = "topic"
    return (
        pd.read_csv(questions_csv_path, usecols=columns)
          .loc[:, list(columns)]
          .rename(columns=columns)
    )


def run_pipeline() -> None:
    """
    Run the FAQ extraction pipeline on the provided dataset.

    Loads the CSV data, processes it, and runs the clustering and
    summarization steps to extract FAQs. Saves the result to a JSON
    file.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    questions_df = load_questions_csv(
        questions_csv_path=args.questions_csv_path,
        questions_column_name=args.questions_column_name,
        answers_column_name=args.answers_column_name,
        topics_column_name=args.topics_column_name,
    )
    questions = questions_df.question.values
    if args.answers_column_name:
        answers = questions_df.answer.values
    else:
        answers = None
    logger.info(f"Loaded {questions.size} questions")

    if args.topics_column_name is not None:
        n_topics = questions_df.topic.nunique()
        kmeans.n_clusters = min(n_topics, args.n_max_clusters)
    elif args.n_clusters is not None:
        kmeans.n_clusters = args.n_clusters
    else:
        kmeans.n_clusters = 25
    logger.info(f"K-Means n clusters set to {kmeans.n_clusters}")

    pipeline = TopNFAQExtractionPipeline(
        n_faqs=args.n_faqs,
        n_example_questions=args.n_example_questions,
        embedding_model=embedding_model,
        dimensionality_reduction_model=pca,
        clustering_model=kmeans,
        summarization_model=summarization_model,
        spelling_correction_model=spelling_correction_model,
        evaluation_metric=args.evaluation_metric,
    )
    extracted_faqs = pipeline(questions=questions, answers=answers)
    with args.output_json_path.open("w") as f:
        json.dump(extracted_faqs, f, indent=4)
    logger.info(f"FAQs written to {args.output_json_path}")


if __name__ == "__main__":
    run_pipeline()
