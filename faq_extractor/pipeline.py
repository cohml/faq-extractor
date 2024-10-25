from collections import Counter
import re
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import pipeline
from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline

from .cli import args
from .utils import logger


FAQ_RE = re.compile(r"FAQ:\s*(?P<faq>.+)")
FAQ_ANSWER_RE = re.compile(r"FAQ:\s*(?P<faq>.+?)\s*ANSWER:\s*(?P<answer>.+)")


class TopNFAQExtractionPipeline:
    """
    Pipeline for extracting the top n most common FAQs from a dataset.

    This pipeline uses embeddings, dimensionality reduction, and
    clustering to identify FAQs, and optionally applies evaluation
    metrics and spelling correction.

    Parameters
    ----------
    n_faqs : int
        Number of most frequent FAQs to extract.
    n_example_questions : int
        Number of example questions to return per FAQ.
    embedding_model : SentenceTransformer
        Model used for embedding the questions.
    dimensionality_reduction_model : PCA
        Model for reducing the embedding dimensionality.
    clustering_model : KMeans
        Clustering model to group questions by semantic similarity.
    summarization_model : Callable
        Model to generate a concise FAQ from the clustered questions.
    spelling_correction_model : Optional[Text2TextGenerationPipeline]
        Model for optional spelling correction before evaluation.
    evaluation_metric : Optional[str]
        Metric for evaluating the quality of the FAQ extractions.
    """

    def __init__(
        self,
        n_faqs: int,
        n_example_questions: int,
        embedding_model: SentenceTransformer,
        dimensionality_reduction_model: PCA,
        clustering_model: KMeans,
        summarization_model: Callable,
        spelling_correction_model: Optional[Text2TextGenerationPipeline],
        evaluation_metric: Optional[str],
    ):
        self.embedding_model = embedding_model
        self.dimensionality_reduction_model = dimensionality_reduction_model
        self.clustering_model = clustering_model
        self.summarization_model = summarization_model
        self.spelling_correction_model = spelling_correction_model
        self.n_faqs = n_faqs
        self.n_example_questions = n_example_questions
        self.evaluation_metric = None
        if evaluation_metric:
            self.evaluation_metric = evaluate.load(evaluation_metric)

    def embed(self, questions: np.ndarray[str]) -> np.ndarray[np.float32]:
        """
        Encode the questions as embedding vectors.

        Parameters
        ----------
        questions : np.ndarray[str]
            Array of questions to embed.

        Returns
        -------
        np.ndarray[np.float32]
            Embedding vectors corresponding to the input questions.
        """
        return self.embedding_model.encode(questions)

    def reduce(
        self,
        embeddings: np.ndarray[np.float32],
    ) -> np.ndarray[np.float32]:
        """
        Reduce the dimensionality of the embedding vectors using PCA.

        Parameters
        ----------
        embeddings : np.ndarray[np.float32]
            High-dimensional embeddings of the questions.

        Returns
        -------
        np.ndarray[np.float32]
            Lower-dimensional embeddings using principal component
            analysis.
        """
        return self.dimensionality_reduction_model.fit_transform(embeddings)

    def cluster(
        self,
        principal_components: np.ndarray[np.float32],
    ) -> np.ndarray[np.int32]:
        """
        Cluster the reduced embeddings using K-Means.

        Parameters
        ----------
        principal_components : np.ndarray[np.float32]
            Dimensionality-reduced question embeddings.

        Returns
        -------
        np.ndarray[np.int32]
            Cluster labels indicating the cluster assigned to each
            question.
        """
        return self.clustering_model.fit_predict(principal_components)

    def get_n_largest_clusters(
        self,
        all_clusters: np.ndarray[np.int32],
        n_faqs: int,
    ) -> np.ndarray[np.int32]:
        """
        Identify the n largest clusters by number of questions.

        Parameters
        ----------
        all_clusters : np.ndarray[np.int32]
            Array of cluster labels for each question.
        n_faqs : int
            Number of largest clusters to return.

        Returns
        -------
        np.ndarray[np.int32]
            Array of the largest cluster labels.
        """
        return [
            cluster for cluster, count in Counter(all_clusters).most_common(n_faqs)
        ]

    def get_top_n_prototypical_questions(
        self,
        cluster: np.int32,
        cluster_principal_components: np.ndarray[np.float32],
        questions: np.ndarray[str],
        n_example_questions: int,
    ) -> np.ndarray[str]:
        """
        Identify the most prototypical questions within a cluster,
        defined as the top n most proximal to the cluster centroid.

        Parameters
        ----------
        cluster : np.int32
            Cluster label for which prototypical questions are sought.
        cluster_principal_components : np.ndarray[np.float32]
            Principal components of the questions in the given cluster.
        questions : np.ndarray[str]
            Array of questions in the given cluster.
        n_example_questions : int
            Number of prototypical questions to return.

        Returns
        -------
        np.ndarray[str]
            Array of prototypical questions from the given cluster.
        """
        cluster_centroid = self.clustering_model.cluster_centers_[cluster]
        distances = np.linalg.norm(
            cluster_principal_components - cluster_centroid, axis=1
        )
        n_closest = abs(distances).argsort()[:n_example_questions]
        return questions[n_closest]

    def summarize_latent_faq(
        self,
        example_questions: np.ndarray[str],
        example_answers: Optional[np.ndarray[str]],
    ) -> Dict[str, str]:
        """
        Extract an FAQ summary from a set of example questions and
        optional answers using GPT.

        Parameters
        ----------
        example_questions : np.ndarray[str]
            Array of example questions from a single cluster.
        example_answers : Optional[np.ndarray[str]]
            Optional array of example answers, if provided.

        Returns
        -------
        Dict[str, str]
            Dictionary containing the extracted FAQ and optional
            answer.
        """
        if example_answers is not None:
            examples = [
                {"question": question, "answer": answer}
                for question, answer in zip(example_questions, example_answers)
            ]
            prompt = (
                "Here are several question-answer pairs one might expect "
                f"to see in a customer service context:\n{examples}\n"
                "Synthesize what these inquiries are asking about, extract "
                "the (1) single underlying question common to most or all of "
                "them and (2) the associated correct answer, and rephrase "
                "both in the simplest, most direct manner possible. Format "
                'your output exactly as "FAQ: {question summary}\nANSWER: '
                '{answer summary}". No other tokens are needed.'
            )
        else:
            prompt = (
                "Here are several questions one might expect to see in a "
                f"customer service context:\n{example_questions}\nSynthesize "
                "what these inquiries are asking about, extract the single "
                "underlying question common to most or all of them, and "
                "rephrase it in the simplest, most direct manner possible. "
                'Format your output exactly as "FAQ: {question summary}". No '
                "other tokens are needed."
            )
        message = [{"role": "user", "content": prompt}]
        n_attempted = 0
        while n_attempted < 3:
            completion = self.summarization_model(messages=message)
            generation = completion.choices[0].message.content
            if example_answers is not None:
                match = FAQ_ANSWER_RE.search(generation)
                if match:
                    faq_dict = match.groupdict()
                    break
            else:
                match = FAQ_RE.search(generation)
                if match:
                    faq_dict = match.groupdict()
                    break
            logger.warning(
                "Unable to parse GPT output. Generating new response."
            )
        if n_attempted == 3:
            logger.error(
                "Unable to parse GPT output. Returning raw string."
            )
            faq_dict = {"faq": generation}
        return faq_dict

    def sample_example_questions(
        self,
        questions: np.ndarray[str],
        answers: Optional[np.ndarray[str]],
        all_clusters: np.ndarray[np.int32],
        n_largest_clusters: np.ndarray[np.int32],
    ) -> Tuple[List[np.ndarray[str]], Optional[List[np.ndarray[str]]]]:
        """
        Take uniformly random samples of example questions and
        optionally answers from the largest clusters.

        Parameters
        ----------
        questions : np.ndarray[str]
            Array of all questions in the dataset.
        answers : Optional[np.ndarray[str]]
            Optional array of answers corresponding to the questions.
        all_clusters : np.ndarray[np.int32]
            Array of cluster labels for all questions.
        n_largest_clusters : np.ndarray[np.int32]
            Array of the largest cluster labels.

        Returns
        -------
        Tuple[List[np.ndarray[str]], Optional[List[np.ndarray[str]]]]
            Arrays of example questions and optional answers sampled
            from the largest clusters.
        """
        example_questions_by_cluster = []
        example_answers_by_cluster = []
        for cluster in n_largest_clusters:
            is_in_cluster = np.argwhere(all_clusters == cluster)
            sample_indices = (args.random_state or np.random).choice(
                np.arange(is_in_cluster.size),
                size=self.n_example_questions,
            )
            example_questions_by_cluster.append(
                questions[is_in_cluster][sample_indices][:, 0]
            )
            if answers is not None:
                example_answers_by_cluster.append(
                    answers[is_in_cluster][sample_indices][:, 0]
                )
        return example_questions_by_cluster, example_answers_by_cluster

    def evaluate_faq(
        self,
        example_questions: np.ndarray[str],
        faq_text: str,
        faq_id: int,
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a generated FAQ using a specified
        evaluation metric.

        Parameters
        ----------
        example_questions : np.ndarray[str]
            Array of example questions from the source dataset.
        faq_text : str
            The FAQ text and optional answer generated by the
            summarization model.
        faq_id : int
            The identifier of the FAQ being evaluated. For logging
            purposes only.

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metric scores.

        Raises
        ------
        ValueError
            If the specified evaluation metric is not supported.
        """
        if self.spelling_correction_model is not None:
            logger.info(
                f"Computing {self.evaluation_metric.name} for FAQ #{faq_id} with "
                "spelling correction"
            )
            example_questions = [
                corrected["generated_text"] for corrected in self.spelling_correction_model(
                    example_questions.tolist()
                )
            ]
        else:
            logger.info(f"Computing {self.evaluation_metric.name} for FAQ #{faq_id}")
        evaluation_metric_kwargs = {
            "predictions": [faq_text],
            "references": [example_questions],
        }
        if self.evaluation_metric.name == "bert_score":
            evaluation_metric_kwargs["lang"] = "en"
        return self.evaluation_metric.compute(
            **evaluation_metric_kwargs
        )

    def __call__(
        self,
        questions: np.ndarray[str],
        answers: Optional[np.ndarray[str]],
    ) -> List[Dict[str, Any]]:
        """
        Run the FAQ extraction pipeline on the input questions and
        optional answers.

        Parameters
        ----------
        questions : np.ndarray[str]
            Array of question texts to extract FAQs from.
        answers : Optional[np.ndarray[str]]
            Array of answer texts corresponding to the questions.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing extracted FAQs and
            evaluation results (if applicable).
        """
        logger.info(
            f"Extracting {self.n_faqs} FAQs from {questions.size:,} "
            "questions"
        )
        start = perf_counter()

        logger.info("Computing question embeddings")
        embeddings = self.embed(questions)

        logger.info("Reducing dimensionality of question embeddings")
        principal_components = self.reduce(embeddings)

        logger.info(f"Creating {self.clustering_model.n_clusters} embeddings clusters")
        all_clusters = self.cluster(principal_components)

        logger.info(f"Identifying {self.n_faqs} most populous embeddings clusters")
        n_largest_clusters = self.get_n_largest_clusters(
            all_clusters, self.n_faqs
        )

        logger.info(
            f"Sampling {self.n_example_questions} from each of the most populous "
            "embeddings clusters"
        )
        example_questions_by_cluster, example_answers_by_cluster = self.sample_example_questions(
            questions, answers, all_clusters, n_largest_clusters
        )

        logger.info("Extracting latent FAQs from most populous embeddings clusters")
        extracted_faqs = []
        for i, example_questions in enumerate(example_questions_by_cluster):
            if len(example_answers_by_cluster) > 0:
                example_answers = example_answers_by_cluster[i]
                faq_dict = self.summarize_latent_faq(
                    example_questions=example_questions,
                    example_answers=example_answers,
                )
                faq_dict["examples"] = [
                    {"question": question, "answer": answer}
                    for question, answer in zip(example_questions, example_answers)
                ]
            else:
                faq_dict = self.summarize_latent_faq(
                    example_questions=example_questions,
                    example_answers=None,
                )
                faq_dict["examples"] = [
                    {"question": question} for question in example_questions
                ]
            if self.evaluation_metric is not None:
                faq_dict[self.evaluation_metric.name] = self.evaluate_faq(
                    example_questions,
                    faq_dict["faq"],
                    faq_id=i + 1,
                )
            extracted_faqs.append(faq_dict)
        stop = perf_counter()
        logger.info(
            f"FAQs extraction runtime: {stop - start:.2f} seconds"
        )
        return extracted_faqs
