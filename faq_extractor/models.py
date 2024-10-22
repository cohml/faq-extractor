from functools import partial
from typing import Any, Callable, Optional

from openai import AzureOpenAI, OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
from transformers import pipeline

from .cli import args
from .utils import logger


def get_device() -> str:
    """
    Detect available computation device for model inference.

    Parameters
    ----------
    None

    Returns
    -------
    str
        "cuda", "mps", or "cpu", depending on available hardware.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_spelling_correction_model(
    do_spelling_correction: bool,
    spelling_correction_model_name: str,
) -> Optional[pipeline]:
    """
    Load the spelling correction model for use when computing metrics.

    Parameters
    ----------
    do_spelling_correction : bool
        Whether spelling correction should be enabled.
    spelling_correction_model_name : str
        Name of the model to use for spelling correction.

    Returns
    -------
    Optional[pipeline]
        HuggingFace pipeline for text2text-generation if spelling
        correction is enabled, else None.
    """
    if args.do_spelling_correction:
        logger.info(
            f"Loading pretrained spelling correction model: "
            + spelling_correction_model_name
        )
        return pipeline(
            "text2text-generation",
            model=args.spelling_correction_model_name,
            device=-1 if device == "cpu" else 0,
            torch_dtype=torch.float16,
        )
    else:
        logger.warning(
            "Skipping spelling correction model. Evaluation metrics "
            "may be artificially low."
        )
        return None


def load_summarization_model(
    config: dict[str, Any],
    use_azure_openai: bool,
) -> Callable:
    """
    Initialize and return the OpenAI client for GPT inference.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary containing API credentials and model
        metadata.
    use_azure_openai : bool
        Whether to connect via Azure OpenAI or directly via OpenAI.

    Returns
    -------
    Callable
        OpenAI chat completion function preloaded with the desired
        model name.

    Raises
    ------
    KeyError
        If required configuration fields are missing.
    """
    logger.info(
        f"Connecting to OpenAI client for model " + config["OPENAI_MODEL"]
    )
    if use_azure_openai:
        for required_field in ["OPENAI_API_VERSION", "OPENAI_ENDPOINT"]:
            if required_field not in config:
                raise KeyError(
                    f"{required_field!r} is missing from {config_json_path}."
                )
        openai_client = AzureOpenAI(
            api_key=config["OPENAI_API_KEY"],
            api_version=config["OPENAI_API_VERSION"],
            azure_endpoint=config["OPENAI_ENDPOINT"],
        )
    else:
        openai_client = OpenAI(api_key=config["OPENAI_API_KEY"])
    return partial(
        openai_client.chat.completions.create,
        model=config["OPENAI_MODEL"],
        temperature=0.2,
    )


device = get_device()

embedding_model = SentenceTransformer(
    args.embedding_model_name, device=device
).eval()

kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)

pca = PCA(n_components=args.n_principal_components)
logger.info(f"PCA n components set to {args.n_principal_components}")

spelling_correction_model = load_spelling_correction_model(
    args.do_spelling_correction,
    args.spelling_correction_model_name,
)

summarization_model = load_summarization_model(
    args.config_json_path,
    args.use_azure_openai,
)
