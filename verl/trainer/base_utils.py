import torch.nn.functional as F
import torch
from collections import Counter
import nltk
from nltk.corpus import stopwords
from typing import List, Tuple
import numpy as np
import string
import re
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

def avg_delta_logprob_on_alternatives(logits: torch.Tensor, sampled_ids: torch.Tensor) -> float:
    """
    Compute average Δ logprob = logp(chosen) - mean(logp(alternatives))
    over all tokens in a rollout.

    Args:
        logits: torch.Tensor [T, V] 
            Model logits for the rollout.
        sampled_ids: torch.Tensor [T] 
            Sampled token ids from the rollout.

    Returns:
        float: average Δ across all tokens
    """
    
    # Log-softmax to get logprobs
    logprobs = torch.log_softmax(logits, dim=-1)  # [T, V]

    T, V = logprobs.shape
    deltas = []

    for t in range(T):
        chosen = sampled_ids[t]
        chosen_logp = logprobs[t, chosen]

        # Mask out chosen token
        alt_mean = logprobs[t, torch.arange(V) != chosen].mean()

        deltas.append((chosen_logp - alt_mean).item())
    try:
        return sum(deltas) / T
    except:
        return 0.0

def check_kl_to_checkpoint(model_new_logits, model_old_logits, loss_mask):
    """
    Compute KL(model_new || model_old) averaged over tokens.
    
    Args:
        model_new: recent AutoModelForCausalLM
        model_old: earlier checkpoint AutoModelForCausalLM
        tokenizer: matching tokenizer
        texts: list of strings (probe set)
        device: "cuda" or "cpu"
    
    Returns:
        kl_score: float (mean KL divergence per token)
    """
    
    

    with torch.no_grad():
        logp_new = F.log_softmax(model_new_logits, dim=-1)
        logp_old = F.log_softmax(model_old_logits, dim=-1)

    # KL(new || old) = sum p_new * (logp_new - logp_old)
    p_new = logp_new.exp()
    # KL per token
    kl = torch.sum(p_new * (logp_new - logp_old), dim=-1)  # (batch, seq_len)
    
    # apply attention mask if provided
    if loss_mask is not None:
        kl = kl * loss_mask  # zero-out pad positions
        denom = loss_mask.sum()
    else:
        denom = kl.numel()

    # avoid div by 0
    denom = max(denom.item(), 1)
    return (kl.sum() / denom).item()

def rigidity_index_from_logits(logits: torch.Tensor, loss_mask: torch.Tensor = None, k: int = 5, eps=1e-12) -> float:
    """
    Compute Rigidity Index from model logits (sampled rollout).
    
    Args:
        logits: torch.Tensor of shape (T, V) or (B, T, V)
        k: number of top tokens to consider for rigidity
        
    Returns:
        float: average rigidity index across tokens (and batch if present)
    """
    
    if logits.dim() == 2:  # (T, V) -> add batch dim
        logits = logits.unsqueeze(0)
        
    # sanity check for bad logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise ValueError("Input logits contain NaN or Inf values")

    # log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)

    # take top-k logprobs
    topk_logprobs, _ = torch.topk(log_probs, k, dim=-1)  # (B, T, k)

    # convert to probs
    topk_probs = topk_logprobs.exp()

    # normalize safely
    denom = topk_probs.sum(dim=-1, keepdim=True).clamp(min=eps)
    topk_probs = topk_probs / denom

    # entropy over top-k (clamp probs to avoid log(0))
    entropy = -(topk_probs * (topk_probs.clamp(min=eps).log())).sum(dim=-1)  # (B, T)

    if loss_mask is not None:
        entropy = entropy * loss_mask

    # normalize by log(k)
    rigidity = 1.0 - (entropy / torch.log(torch.tensor(float(k), dtype=logits.dtype, device=logits.device)))
    # mean over batch & tokens
    return rigidity.mean(dim=-1).sum(dim=-1).item()


def extract_words(text: str, remove_stopwords: bool = False) -> set:
    """
    Extract words from text.
    
    Args:
        text: input text string
        remove_stopwords: whether to remove common stopwords
        
    Returns:
        set of words (lowercased)
    """
    # Tokenize text (simple word-level tokenization)
    tokens = text.lower().split()
    
    if remove_stopwords:
        try:
            # Download stopwords if not already present
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        english_stopwords = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in english_stopwords]
    
    return set(tokens)


def remove_common_words(query: str, solutions: List[str]) -> List[set]:
    """
    Remove common words between query and solutions from all solutions.
    Optionally also removes stopwords.
    
    Args:
        query: the query string
        solutions: list of solution strings
        
    Returns:
        list of solutions with common words (and optionally stopwords) removed
    """
    query = query.lower().translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    
    # Extract words from query (excluding stopwords if specified)
    query_words = extract_words(query, remove_stopwords=True)  # Don't remove stopwords here, handle separately
    
    cleaned_solutions = []
    for solution in solutions:
        solution = solution.lower().translate(str.maketrans('', '', string.punctuation)) #remove punctuation
        solution_tokens = extract_words(solution, remove_stopwords=True)
        
        # Remove query words and optionally stopwords from solution
        filtered_tokens = set()

        for token in solution_tokens:
            # Skip if it's a query word or a stopword
            if token not in query_words:
                filtered_tokens.add(token)
        
        cleaned_solutions.append(filtered_tokens)
    
    return cleaned_solutions


def pairwise_bleu_score(solutions: List[List[int]]) -> tuple[float, float]:
    """
    Calculate pairwise BLEU scores between all solutions.
    
    Args:
        solutions: list of solutions, each a list of token IDs (or words)
    
    Returns:
        (min_bleu, mean_bleu) across all pairs
    """
    if len(solutions) < 2:
        return 0.0, 0.0
    
    smoothie = SmoothingFunction().method1
    all_scores = []
    
    for i, candidate in enumerate(solutions):
        for j, reference in enumerate(solutions):
            if i != j:
                score = sentence_bleu([list(reference)], list(candidate), smoothing_function=smoothie)
                all_scores.append(score)
    
    return float(np.min(all_scores)), float(np.mean(all_scores))


def type_token_ratio(solutions: List[List[int]]) -> float:
    """
    Compute Type-Token Ratio (TTR) as a measure of diversity across rollouts.
    
    Args:
        solutions: list of solutions, each a list of token IDs (or words)
    
    Returns:
        float: ratio of unique tokens to total tokens across all solutions
    """
    all_tokens = [tok for seq in solutions for tok in seq]
    if not all_tokens:
        return 0.0
    unique_tokens = set(all_tokens)
    return len(unique_tokens) / len(all_tokens)


def jaccard_similarity(words1: set, words2: set) -> float:
    """
    Calculate Jaccard similarity between two texts.
    Jaccard = |intersection| / |union| of word sets
    
    Args:
        text1: first text string
        text2: second text string
        
    Returns:
        float: Jaccard similarity score (0.0 to 1.0)
    """
    
    # Calculate Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def similarity_score_cal(query: str, solutions: List[str]) -> Tuple[float, float, float, float, float]:
    """
    Process query and solutions by removing common words and calculating average Jaccard similarity.
    
    Args:
        query: the query string
        solutions: list of solution strings
        
    Returns:
        tuple of (cleaned_solutions, average_jaccard_similarity)
    """
    # Remove common words between query and solutions (and optionally stopwords)
    cleaned_solutions = remove_common_words(query, solutions)
    
    smoothie = SmoothingFunction().method1
    
    if len(solutions) < 2:
        return 0.0
    
    all_scores = []
    bleu_scores = []
    ttr_score = type_token_ratio(cleaned_solutions)
    for i, candidate in enumerate(cleaned_solutions):
        for j, reference in enumerate(cleaned_solutions):
            if i != j:  # Don't compare solution with itself
                jaccard_score = jaccard_similarity(candidate, reference)
                bleu_score = sentence_bleu([list(reference)], list(candidate), smoothing_function=smoothie)
                bleu_scores.append(bleu_score)
                all_scores.append(jaccard_score)
    
    return np.min(all_scores).item(), np.mean(all_scores).item(), ttr_score, np.mean(bleu_scores).item(), np.min(bleu_scores).item()
    

