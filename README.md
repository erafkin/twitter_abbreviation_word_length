# Surprisal Theory on Compressions on the Internet
Code for generating data and analysis for surprisal theory on Tweets for abbreviations

## Repo Structure:
- [`notebooks/`](./notebooks/)
    - [`explore.ipynb`](./notebooks/explore.ipynb) generates the analyses used in the paper
    - [`tweet_length.ipynb`](./notebooks/tweet_length.ipynb) looks into the distribution of the keywords (long and short forms) based in tweet length to identify whether more abbreviations are used need the Twitter character limit
    -[`tweets.ipynb`](./notebooks/tweets.ipynb) mainly just a playground to see how to run through the data and find the relevant tweets.
    - [`word_length.ipynb`](./notebooks/word_length.ipynb) compares Twitter to Wikipedia to determine if words are shorter on Twitter than in other domains. 
- [`scripts`](./scripts/)
    - [`utils`](./scripts/utils/)
        - [`abbreviations.py`](./scripts/utils/abbreviations.py) dictionary of the long & short form pairs
        - [`surprisal.py`](./scripts/utils/surprisal.py) calculates surprisal for phrases in a tweet. Both [PLL](https://aclanthology.org/2020.acl-main.240/) and [PLL-l2r](https://aclanthology.org/2023.acl-short.80.pdf) are calculated
        - [`TweetNormalizer.py`](./scripts/utils/TweetNormalizer.py) BERTweet normalization methods taken from the [BERTweet repository](https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py)
    - [`get_data.py`](./scripts/get_data.py) actually creates the dataframe from the models and data
- [`data_df.csv`](./data_df.csv) CSV of the data of the counts and surprisal for long and short forms of different meanings
- [`tweets_containing_words_and_full_length.csv`](./tweets_containing_words_and_full_length.csv) CSV of keywords/keyphrases and the tweets containing them. 

## Data & Model
- [Cardiff NLP Tweet Eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [Engilish WikiDump](https://huggingface.co/datasets/legacy-datasets/wikipedia)
- [BERTweet](https://huggingface.co/docs/transformers/en/model_doc/bertweet)

## setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Notes
developed in python 3.11.6

## Author
Emma Rafkin `epr41@georgetown.edu`
