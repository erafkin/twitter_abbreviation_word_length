import pandas as pd
import numpy as np
from scripts.utils.surprisal import calculate_abbreviation_surprisal, calculate_phrase_surprisal
from scripts.utils.abbreviations import abbrevations
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def get_calcuations(abbreviation_dict, tokenizer, model, dataset):
    df_arr = []
    for full_length, abbr in tqdm(abbreviation_dict.items()):
        if  tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(abbr)) == tokenizer.unk_token:
            print("abbreviation unknown: ", abbr)
            pass
        count_fl = 0
        count_abbr = 0
        tweets_with_abbr = []
        tweets_with_fl = []
        fl_tokenized = tokenizer.tokenize(full_length)
        abbr_tokenized =  tokenizer.tokenize(abbr)
        for split in dataset:
            for i in range(len(dataset[split])):
                tokens = tokenizer.tokenize(dataset[split][i]["text"])
                tokens = [t.lower() for t in tokens]
                tokens = [t.replace("'", "") for t in tokens]
                fl_in_tweet = [tokens[idx: idx + len(fl_tokenized)] == fl_tokenized for idx in range(len(tokens) - len(fl_tokenized) + 1)].count(True)
                abbr_in_tweet = [tokens[idx: idx + len(abbr_tokenized)] == abbr_tokenized for idx in range(len(tokens) - len(abbr_tokenized) + 1)].count(True)

                if fl_in_tweet > 0:
                    count_fl += fl_in_tweet
                    tweets_with_fl.append(dataset[split][i]["text"])
                if abbr_in_tweet > 0:
                    count_abbr+= abbr_in_tweet
                    tweets_with_abbr.append(dataset[split][i]["text"])

        abbr_surprisals = np.array([calculate_abbreviation_surprisal(model, tokenizer=tokenizer, tweet=tweet, word=abbr) for tweet in tweets_with_abbr])
        abbr_surprisals = abbr_surprisals[~pd.isnull(abbr_surprisals)]
        full_length_surprisals = np.array([calculate_phrase_surprisal(model, tokenizer=tokenizer, tweet=tweet, phrase=full_length) for tweet in tweets_with_fl])
        full_length_surprisals = full_length_surprisals[~pd.isnull(full_length_surprisals)]
        df_arr.append([abbr, full_length, count_abbr, count_fl, np.mean(abbr_surprisals), np.mean(full_length_surprisals)])
    
    df = pd.DataFrame(df_arr, columns=["abbreviation", "full_length", "count_abbr", "count_fl", "surprisal_abbr", "surprisal_fl"])
    return df

if __name__ == "__main__":
    subset={
        "brother":"bro", 
        "oh my god": "omg",
        "sorry": "sry", 
        "about": "abt",
        "direct message": "dm",
        "retweet": "rt"
        }
    bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    df = get_calcuations(abbreviation_dict=subset, tokenizer=tokenizer, model=bertweet, dataset=ds)
    df.to_csv("./test.csv", index=False)
    # df.to_csv("./data_df.csv", index=False)
