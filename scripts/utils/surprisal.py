import torch
import numpy as np
from scripts.utils.TweetNormalizer import normalizeTweet

# TODO: Think about combined liklihood for the representation as a whole.

def calculate_abbreviation_surprisal(model, tokenizer, tweet, word):
    # calculate surprisal for a given word based on a tweet
    # NOTE: This is just for a single word
    t = tweet.lower()
    words = tokenizer.tokenize(t)
    
    line = " ".join(words)
    normalized_line = normalizeTweet(tweet=line)
    split_line = normalized_line.split()
    mask_ids = [i for i in range(len(split_line)) if split_line[i] == word]
    for mask_id in mask_ids:
        split_line[mask_id] = tokenizer.mask_token
    normalized_line = " ".join(split_line)
    
    input_ids = torch.tensor([tokenizer.encode(normalized_line)])
    logits = model(input_ids).logits
    surprials = []
    for mask_id in mask_ids:
        mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].numpy()[0]
        probs =  torch.nn.functional.softmax(logits[0, mask_idx, :], dim=0)
        word_idx = tokenizer.convert_tokens_to_ids(word)
        surprisal = -torch.log2(probs[word_idx]).detach().numpy()
        surprials.append(surprisal)
    return np.mean(surprials)
    
def calculate_phrase_surprisal(model, tokenizer, tweet, phrase):
    t = tweet.lower()
    words = tokenizer.tokenize(t)
    tokenized_phrase = tokenizer.tokenize(phrase)
    line = " ".join(words)
    normalized_line = normalizeTweet(tweet=line)
    split_line = normalized_line.split()
    mask_ids = []
    for phrase_token in tokenized_phrase:
        phrase_mask_ids = [i for i in range(len(split_line)) if split_line[i] == phrase_token]
        for mask_id in phrase_mask_ids:
            split_line[mask_id] = tokenizer.mask_token
        mask_ids.append(phrase_mask_ids)
    normalized_line = " ".join(split_line)
    input_ids = torch.tensor([tokenizer.encode(normalized_line)])
    logits = model(input_ids).logits
    surprials = []
    for phrase_mask_ids in mask_ids:
        phrase_surprisals = []
        for idx, mask_id in enumerate(phrase_mask_ids):
            mask_idx = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].numpy()[0]
            probs =  torch.nn.functional.softmax(logits[0, mask_idx, :], dim=0)
            # WIP
            word_idx = tokenizer.convert_tokens_to_ids(tokenized_phrase[idx])
            surprisal = -torch.log2(probs[word_idx]).detach().numpy()
            phrase_surprisals.append(surprisal)
        surprials.append(np.mean(phrase_surprisals))
    return np.mean(surprials)
