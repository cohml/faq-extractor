# FAQ Extractor

`faq_extractor` is a Python package designed to extract the top frequently
asked questions (FAQs) from a dataset of questions and answers, such as
customer service logs.

This package leverages a pipeline of language models and machine learning
algorithms to perform dimensionality reduction, clustering, and summarization.

It is primarily intended to be used through a command-line utility
(`faq-extractor`) installed in a conda environment. However, the main file
`main.py` can also be executed directly with the same effect.

## Overview

The `faq_extractor` package allows users to identify common FAQs from large
datasets of user-submitted questions, automatically grouping related queries
and summarizing them into distinct FAQs. It is particularly useful for
customer support teams, chatbot training, and knowledge base management, where
large volumes of similar questions need to be distilled into concise,
representative FAQs.

### Key Features:

- Clusters semantically similar questions in lower-dimensional embedding space.
- Summarizes representative FAQs and optional answers using GPT.
- Optional spelling correction and metric evaluation (e.g., BERTScore, ROUGE).
- Easily configurable via command-line arguments.
- Intended to work with both OpenAI and Azure OpenAI GPT deployments.

> **Important:**
> The author of this package uses Azure OpenAI. As a result, the code path
> executed when using OpenAI (not Azure OpenAI) could not be tested.

## Installation

To install the package and its dependencies as appropriate for your operating
system, run the following script from the root directory of the project:

```bash
bash setup-env.sh
```

This will create a new conda environment named `faq-extractor` with all
required dependencies. The environment includes a `faq-extractor`
command-line utility, which can be used to interact with the package.

If you experience `numpy` errors during or after installation, try installing
`pytorch` from `conda-forge` instead of the conventional `pytorch` conda
channel.

## How It Works

The FAQ extraction pipeline uses the following components:

1. **Sentence Embedding Model**: Questions are embedded into dense vector
   representations using a pretrained sentence transformer model (default:
   `paraphrase-MiniLM-L6-v2`).

2. **Dimensionality Reduction (PCA)**: High-dimensional embeddings are reduced
   to a lower-dimensional space using Principal Component Analysis (PCA) for
   more efficient and less noisy clustering.

3. **Clustering (K-Means)**: The reduced embeddings are clustered using
   K-Means. Each cluster represents a group of semantically similar questions,
   which can later be summarized into an FAQ.

4. **Summarization (GPT)**: Using OpenAI’s GPT (or Azure OpenAI), a
   representative FAQ is extracted from each cluster, optionally accompanied
   by an answer if available.

5. **Evaluation**: If specified, the quality of the extracted FAQs can be
   evaluated using metrics such as BERTScore or ROUGE. Optionally, the
   questions can undergo spelling correction before evaluation.

## Command-Line Utility

`faq_extractor` is primarily accessed via the `faq-extractor` CLI tool, which
processes a CSV file containing questions and outputs the extracted FAQs to a
JSON file. Running it is equivalent to executing the main file `main.py`.

### Example Command

```bash
faq-extractor --questions-csv-path my_data.csv --config-json-path ./config.json
```

> **Tip:**
> To simply get a quick feel for what `faq-extractgor` will generate,
> note that a Q&A dataset is already provided as the default argument for
> `--questions-csv-path`. So that argument can be safely omitted when quickly
> experimenting.

### Command-Line Arguments

- `--questions-csv-path`: Path to the CSV file containing the dataset of questions. Default: `hf://datasets/bitext/...`.
- `--questions-column-name`: Column name in the CSV file containing the questions. Default: `instruction`.
- `--answers-column-name`: (Optional) Column name containing the corresponding answers.
- `--topics-column-name`: (Optional) Column name containing topic/intent labels. Helps set the optimal number of clusters.
- `--output-json-path`: Path to the output JSON file where extracted FAQs will be saved. Default: `faqs.json`.
- `--config-json-path`: Path to the configuration JSON file required to connect to GPT/OpenAI APIs. **Required**.
- `--n-faqs`: Number of most frequent FAQs to extract. Default: `5`.
- `--n-example-questions`: Number of example questions to include per extracted FAQ. Default: `5`.
- `--embedding-model-name`: Name of the embedding model (compatible with `sentence-transformers`). Default: `paraphrase-MiniLM-L6-v2`.
- `--spelling-correction-model-name`: Name of the HuggingFace model for spelling correction. Default: `oliverguhr/spelling-correction-english-base`.
- `--n-principal-components`: Number of dimensions to project semantic vectors onto before clustering. Default: `50`.
- `--n-clusters`: Number of clusters for FAQ extraction. If unspecified, inferred from the data or set to `25`.
- `--n-max-clusters`: Maximum number of clusters for clustering. Default: `25`.
- `--evaluation-metric`: Metric for evaluating extracted FAQs (`bertscore` or `rouge`). If unspecified, no evaluation is performed.
- `--do-spelling-correction`: Flag to enable spelling correction before evaluation.
- `--use-azure-openai`: Flag indicating whether GPT is deployed via Azure OpenAI.
- `--temperature`: Temperature setting for GPT generation stage. Default: `None`.
- `--seed`: Seed value for reproducibility. Default: `None`.

To see this information on the terminal, run the following command:

```bash
faq-extractor --help
```

## Configuration File

The `config.json` file contains credentials and model information for
interacting with OpenAI's GPT. The required fields are:

```json
{
  "OPENAI_API_KEY": "your-api-key",
  "OPENAI_MODEL": "gpt-3.5-turbo",
}
```

If interacting with GPT via Azure OpenAI, the following fields are required in
addition:

```
  "OPENAI_API_VERSION": "2023-05-15",
  "OPENAI_ENDPOINT": "https://api.openai.com/v1/engines/"
}
```

## Example Workflow

1. Prepare a dataset in CSV format, ensuring that the column names match those
   specified in the command-line arguments.
2. Create a `config.json` file containing the necessary API credentials for
   GPT access.
3. Run the `faq-extractor` command with the appropriate arguments.
4. The extracted FAQs will be saved to the specified JSON file.
