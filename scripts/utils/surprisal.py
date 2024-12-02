import torch
import numpy as np
from scripts.utils.TweetNormalizer import normalizeTweet

MAX_LENGTH = 128
def calculate_abbreviation_surprisal(model, tokenizer, tweet, word):
    # calculate surprisal for a given word based on a tweet
    t = tweet.lower()
    normalized_line = normalizeTweet(tweet=t)
    words = tokenizer.tokenize(normalized_line)

    mask_ids = [i for i in range(len(words)) if words[i] == word]    
    surprisals = []
    for mask_id in mask_ids:
        words[mask_id] = tokenizer.mask_token
        input_ids = tokenizer.encode(tokenizer.convert_tokens_to_string(words), padding="max_length", max_length=MAX_LENGTH)
        if len(input_ids) > MAX_LENGTH:
            min_idx = mask_ids[0]-int(MAX_LENGTH/2)
            if min_idx<0:
                min_idx=0
            max_idx = mask_ids[0]+int(MAX_LENGTH/2)
            if max_idx>len(input_ids):
                max_idx = len(input_ids)-1
            input_ids = input_ids[min_idx:max_idx]
        logits = model(torch.tensor([input_ids])).logits
        probs =  torch.nn.functional.softmax(logits[0, mask_id, :], dim=0)
        word_idx = tokenizer.convert_tokens_to_ids(word)
        surprisal = -torch.log2(probs[word_idx]).detach().numpy() # why is this sometimes nan?
        surprisals.append(surprisal)
    return np.mean(surprisals)
    
def calculate_phrase_surprisal(model, tokenizer, tweet, phrase):
    t = tweet.lower()
    normalized_line = normalizeTweet(tweet=t)
    words = tokenizer.tokenize(normalized_line)
    tokenized_phrase = tokenizer.tokenize(phrase)
    mask_ids = []
    temp_phrase_mask_ids = [i for i in range(len(words)) if words[i] == tokenized_phrase[0]]
    phrase_mask_ids = []
    for pmi in temp_phrase_mask_ids:
        full_phrase_in_tweet = True
        phrase_masks = []
        for idx, token in enumerate(tokenized_phrase):
            phrase_masks.append(pmi + idx)
            try:
                if words[pmi + idx] != token:
                    full_phrase_in_tweet = False
            except Exception as e:
                print("there was an exception")
                print(e)
                print(pmi, words, phrase)
                full_phrase_in_tweet = False
        if full_phrase_in_tweet:
            phrase_mask_ids.append(phrase_masks)
    surprisals = []
    for phrase_mask_group in phrase_mask_ids:
        phrase_surprisals = []
        for pmi in phrase_mask_group:
            words[pmi] = tokenizer.mask_token
            input_ids = tokenizer.encode(tokenizer.convert_tokens_to_string(words), padding="max_length", max_length=MAX_LENGTH)
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
        surprisals.append(np.mean(phrase_surprisals))
    return np.mean(surprisals)
