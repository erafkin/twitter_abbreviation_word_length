import torch
import numpy as np
from scripts.utils.TweetNormalizer import normalizeTweet

MAX_LENGTH = 128
    
def calculate_phrase_surprisal(model, tokenizer, tweet, phrase):
    t = tweet.lower()
    normalized_line = normalizeTweet(tweet=t)
    words = tokenizer.tokenize(normalized_line)
    normalized_phrase = normalizeTweet(tweet=phrase)
    tokenized_phrase = tokenizer.tokenize(normalized_phrase)
    mask_ids = []
    temp_phrase_mask_ids = [i for i in range(len(words)) if words[i] == tokenized_phrase[0]]
    phrase_mask_ids = []
    for pmi in temp_phrase_mask_ids:
        full_phrase_in_tweet = True
        phrase_masks = []
        for idx, token in enumerate(tokenized_phrase):
            t = token
            if "'" in t:
                t = t.replace("'", "")
            
            phrase_masks.append(pmi + idx)
            if pmi + idx < len(words):
                try:
                    pt = words[pmi + idx]
                    if "'" in pt:
                        pt = pt.replace("'", "")
                    if pt != t:
                        full_phrase_in_tweet = False
                except Exception as e:
                    print("there was an exception")
                    print(e)
                    print(pmi, idx, words, phrase)
                    full_phrase_in_tweet = False
        if full_phrase_in_tweet:
            phrase_mask_ids.append(phrase_masks)
    mask_l2r_surprisals = []
    sequential_mask_surprisals = []

    for phrase_mask_group in phrase_mask_ids:
        # first mask all tokens in the phrase (also works for a mono token situation like the abbreviations themselves)
        phrase_surprisals = []
        for idx, pmi in enumerate(phrase_mask_group):
            words1 = words[:]
            for p in phrase_mask_group[idx:]:
                try:
                    words1[p] = tokenizer.mask_token
                    # get input ids and truncate if necessary
                    input_ids = tokenizer.encode(tokenizer.convert_tokens_to_string(words1), padding="max_length", max_length=MAX_LENGTH)
                    if len(input_ids) > MAX_LENGTH:
                        min_idx = phrase_mask_group[0] - int(MAX_LENGTH/2)
                        if min_idx<0:
                            min_idx=0
                        max_idx = phrase_mask_group[0]+int(MAX_LENGTH/2)
                        if max_idx>len(input_ids):
                            max_idx = len(input_ids)-1
                        input_ids = input_ids[min_idx:max_idx]
                    # get logits from model
                    logits = model(torch.tensor([input_ids])).logits

                    # get prob for the specific phrase token
                    probs =  torch.nn.functional.softmax(logits[0, pmi, :], dim=0)
                    word_idx = tokenizer.convert_tokens_to_ids(tokenized_phrase[idx])
                    surprisal = -torch.log2(probs[word_idx]).detach().numpy()
                    
                    phrase_surprisals.append(surprisal)
                except Exception as e:
                    print(e)
                    print("words: ", words)
                    print("index that is crashing: ", p)
            
        mask_l2r_surprisals.append(phrase_surprisals)

        #now do the sequential mapping if possible
        if len(tokenized_phrase) == 1:
            # it is all the same
            sequential_mask_surprisals = mask_l2r_surprisals
        else:
            phrase_surprisals = []
            for pmi in phrase_mask_group:
                # mask just the one token, leave the rest
                words1 = words[:] 
                try:
                    words1[pmi] = tokenizer.mask_token
                    input_ids = tokenizer.encode(tokenizer.convert_tokens_to_string(words1), padding="max_length", max_length=MAX_LENGTH)
                    if len(input_ids) > MAX_LENGTH:
                        min_idx = mask_ids[0][0] - int(MAX_LENGTH/2)
                        if min_idx<0:
                            min_idx=0
                        max_idx = mask_ids[0][0]+int(MAX_LENGTH/2)
                        if max_idx>len(input_ids):
                            max_idx = len(input_ids)-1
                        input_ids = input_ids[min_idx:max_idx]
                    logits = model(torch.tensor([input_ids])).logits
                    probs =  torch.nn.functional.softmax(logits[0, pmi, :], dim=0)
                    word_idx = tokenizer.convert_tokens_to_ids(tokenized_phrase[0])
                    surprisal = -torch.log2(probs[word_idx]).detach().numpy()
                    phrase_surprisals.append(surprisal)
                except Exception as e:
                    print(e)
                    print("words: ", words)
                    print("index that is crashing: ", pmi)

                
                
            sequential_mask_surprisals.append(phrase_surprisals)      
    return  mask_l2r_surprisals, sequential_mask_surprisals
