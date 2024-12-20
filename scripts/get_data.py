import pandas as pd
import numpy as np
from scripts.utils.surprisal import calculate_phrase_surprisal
from scripts.utils.abbreviations import abbrevations
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from scripts.utils.TweetNormalizer import normalizeTweet



def get_calcuations(abbreviation_dict, tokenizer, model, datasets):
    df_arr = []
    for full_length, abbr in tqdm(abbreviation_dict.items()): 
        if  tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(abbr)) == tokenizer.unk_token:
            print("abbreviation unknown: ", abbr)
            pass
        print("abbreviation: ", abbr)
        count_fl = 0
        count_abbr = 0
        tweets_with_abbr = []
        tweets_with_fl = []
        fl_normalized = normalizeTweet(full_length)
        fl_tokenized = tokenizer.tokenize(fl_normalized)
        abbr_normalized = normalizeTweet(abbr)
        abbr_tokenized =  tokenizer.tokenize(abbr_normalized)
        for dataset in datasets:
            for split in dataset:
                for i in range(len(dataset[split])):
                    normalized_line = normalizeTweet(tweet=dataset[split][i]["text"])
                    normalized_line = normalized_line.lower()
                    normalized_line = normalized_line.replace("'", "")
                    tokens = tokenizer.tokenize(normalized_line)
                    tokens = [t.lower() for t in tokens]
                    tokens = [t.replace("'", "") for t in tokens]
                    fl_tokenized = [t.lower() for t in fl_tokenized]
                    fl_tokenized = [t.replace("'", "") for t in fl_tokenized]
                    abbr_tokenized = [t.lower() for t in abbr_tokenized]
                    abbr_tokenized = [t.replace("'", "") for t in abbr_tokenized]
                    fl_in_tweet = [tokens[idx: idx + len(fl_tokenized)] == fl_tokenized for idx in range(len(tokens) - len(fl_tokenized) + 1)].count(True)
                    abbr_in_tweet = [tokens[idx: idx + len(abbr_tokenized)] == abbr_tokenized for idx in range(len(tokens) - len(abbr_tokenized) + 1)].count(True)

                    if fl_in_tweet > 0:
                        count_fl += fl_in_tweet
                        tweets_with_fl.append(dataset[split][i]["text"])
                    if abbr_in_tweet > 0:
                        count_abbr+= abbr_in_tweet
                        tweets_with_abbr.append(dataset[split][i]["text"])
        abbr_mask_l2r_avg_per_token= []
        abbr_seq_avg_per_token =[]
        abbr_surprisals_mask_l2r= []
        abbr_surprisals_sequential= []
        for tweet in tqdm(tweets_with_abbr):
            ma, seq = calculate_phrase_surprisal(model, tokenizer=tokenizer, tweet=tweet, phrase=abbr)
            abbr_surprisals_mask_l2r += ma
            abbr_surprisals_sequential += seq
        abbr_mask_l2r_avg_per_token = np.array([np.average(s) for s in abbr_surprisals_mask_l2r])
        abbr_seq_avg_per_token = np.array([np.average(s) for s in abbr_surprisals_sequential])
        abbr_surprisals_mask_l2r = np.array([np.sum(s) for s in abbr_surprisals_mask_l2r])
        abbr_surprisals_sequential = np.array([np.sum(s) for s in abbr_surprisals_sequential])

        fl_mask_l2r_avg_per_token= []
        fl_seq_avg_per_token= []
        fl_surprisals_mask_l2r= []
        fl_surprisals_sequential= []
        for tweet in tqdm(tweets_with_fl):
            ma, seq = calculate_phrase_surprisal(model, tokenizer=tokenizer, tweet=tweet, phrase=full_length)
            fl_surprisals_mask_l2r += ma
            fl_surprisals_sequential += seq
        fl_mask_l2r_avg_per_token = np.array([np.average(s) for s in fl_surprisals_mask_l2r])
        fl_seq_avg_per_token = np.array([np.average(s) for s in fl_surprisals_sequential])
        fl_surprisals_mask_l2r = np.array([np.sum(s) for s in fl_surprisals_mask_l2r])
        fl_surprisals_sequential = np.array([np.sum(s) for s in fl_surprisals_sequential])
        df_arr.append([abbr, 
                       full_length, 
                       count_abbr, 
                       count_fl, 
                       np.mean(abbr_mask_l2r_avg_per_token),
                       np.mean(abbr_seq_avg_per_token),    
                       np.mean(abbr_surprisals_mask_l2r), 
                       np.mean(abbr_surprisals_sequential), 
                       np.mean(fl_mask_l2r_avg_per_token),
                       np.mean(fl_seq_avg_per_token),             
                       np.mean(fl_surprisals_mask_l2r), 
                       np.mean(fl_surprisals_sequential), 
                     ])
    
    return df_arr

if __name__ == "__main__":
    configs = ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']
    # configs = ["emotion"]
    bertweet = AutoModelForMaskedLM.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    datasets = [load_dataset("cardiffnlp/tweet_eval", config) for config in configs]
    subset={"oh my god": "omg"}
    df_arr = get_calcuations(abbreviation_dict=abbrevations, tokenizer=tokenizer, model=bertweet, datasets=datasets)
    df = pd.DataFrame(df_arr, columns=["abbreviation", 
                                       "full_length", 
                                       "count_abbr", 
                                       "count_fl", 
                                       "abbr_mask_l2r_avg_per_token",
                                       "abbr_seq_avg_per_token", 
                                       "abbr_mask_l2r_surprisal_mean", 
                                       "abbr_sequential_surprisal_mean",
                                       "fl_mask_l2r_avg_per_token", 
                                       "fl_seq_avg_per_token",
                                       "fl_mask_l2r_surprisal_mean", 
                                       "fl_sequential_surprisal_mean"])
    df.to_csv("data_fix_PLL.csv", index=False)
