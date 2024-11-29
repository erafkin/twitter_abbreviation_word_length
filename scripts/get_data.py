import pandas as pd
import numpy as np
from scripts.utils.surprisal import calculate_abbreviation_surprisal
from scripts.utils.abbreviations import abbrevations
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset


def get_calcuations(abbreviation_dict, tokenizer, model, dataset):
    df_arr = []
    for full_length, abbr in abbreviation_dict.items():
        if  tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(abbr)) == tokenizer.unk_token:
            print("abbreviation unknown: ", abbr)
            pass
        count_fl = 0
        count_abbr = 0
        tweets_with_abbr = []
        tweets_with_fl = []
        for split in dataset:
            for i in range(len(dataset[split])):
                tokens = dataset[split][i]["text"].split()
                tokens = [t.lower() for t in tokens]
                tokens = [t.replace("'", "") for t in tokens]
                if " " in full_length:
                    if abbr in " ".join(tokens):
                        count_fl += 1
                        tweets_with_fl.append(dataset[split][i]["text"])
                else:
                    if full_length in tokens:
                        count_fl += 1
                        tweets_with_fl.append(dataset[split][i]["text"])
                if abbr in tokens:
                    count_abbr+=len([t for t in tokens if t == abbr])
                    tweets_with_abbr.append(dataset[split][i]["text"])
        abbr_surprisals = [calculate_abbreviation_surprisal(model, tokenizer=tokenizer, tweet=tweet, word=abbr) for tweet in tweets_with_abbr]
        full_length_surprisals = [] #TODO we want to calculate surprisal also using the big word
        if count_abbr > 0 and count_fl > 0:
            df_arr.append([abbr, full_length, count_abbr, count_fl, np.mean(abbr_surprisals)])
    
    df = pd.DataFrame(df_arr, columns=["abbreviation", "full_length", "count_abbr", "count_fl", "surprisal_abbr"])
    return df

if __name__ == "__main__":
    bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    ds = load_dataset("cardiffnlp/tweet_eval", "emoji")
    df = get_calcuations(abbreviation_dict=abbrevations, tokenizer=tokenizer, model=bertweet, dataset=ds)
