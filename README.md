# arxiv-data-exploration

Exploratory data analysis and tooling on ArXiv (https://arxiv.org/) Metadata and Data. 

## Dataset

[ArXiv](https://arxiv.org/) [publicly exposes](https://info.arxiv.org/help/api/basics.html) it's metadata and data (the actual PDF files) on it's scholarly articles for anyone to download and extract useful information as needed. You can figure what are the most cited articles and authors, what are the field defining articles, find trends in research, etc. with it.

Instead of directly using ArXiv public API, I have instead chosen to use [Kaggle's compiled metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv) on more than 2.5 million scholarly articles. The snapshot itself is updated every week, so, you should have updated information every week.

## Explorations

### Trend Exploration

**Data**: To start with any notebook, you must download the dataset available on [Kaggle's website](https://www.kaggle.com/datasets/Cornell-University/arxiv/data) alongside the notebook.

**Notebook**: [*arxiv_data_exploration.ipynb*](./trend_exploration/arxiv_trend_exploration.ipynb)
