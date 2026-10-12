"""
PGS/PGSS: Perplexity-Guided Sampling

Computes per-prefix perplexity using Quick-Answer Prompting (QAP) + prefix token approximation.
Used by PGS (pure PPL ordering) and PGSS (Bandit + PPL ordering) strategies.
"""

import numpy as np
import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def _build_qap_prompt(original_chat, quick_prompt):
    """Apply Quick-Answer Prompting to the original chat messages.

    For countdown-style prompts where the last part contains
    "Let me solve this step by step.", replace it with the quick_prompt.

    Args:
        original_chat: list of chat message dicts with 'role' and 'content',
                       or a single message array from parquet
        quick_prompt: str, the quick-answer prompt to use

    Returns:
        Modified prompt string
    """
    # The dataset stores chat as a numpy array of message dicts
    if hasattr(original_chat, '__len__') and len(original_chat) > 0:
        content = original_chat[0]['content']
    elif isinstance(original_chat, str):
        content = original_chat
    else:
        content = str(original_chat)

    # Common assistant prefixes that we want to replace with the quick prompt
    replacements = [
        "Let me solve this step by step.",
        "Let me think step by step.",
    ]
    for old in replacements:
        if old in content:
            return content.replace(old, quick_prompt)

    # If no known prefix found, look for assistant marker and replace after it
    for marker in ["Assistant:", "assistant:"]:
        if marker in content:
            parts = content.split(marker)
            if len(parts) >= 2:
                return parts[0] + marker + " " + quick_prompt

    # Fallback: append quick_prompt as an extra instruction
    return content + "\n" + quick_prompt


def compute_prefix_entropy(actor_rollout_wg, dataset, tokenizer, ppl_config):
    """Compute per-prefix perplexity for a sampled subset of the dataset using QAP.

    At each curriculum update interval:
    1. Sample `sample_size` indices from dataset
    2. Apply Quick-Answer Prompting to modify prompts
    3. Generate short prefixes (prefix_length tokens) via rollout worker
    4. Compute log-probs via actor forward pass
    5. Per-prefix PPL = exp(-mean(log P(y_t | y_{<t}, x'))) for valid prefix tokens

    Args:
        actor_rollout_wg: the actor rollout worker group
        dataset: RLHFDataset to sample from
        tokenizer: the tokenizer
        ppl_config: config with prefix_length, sample_size, quick_prompt

    Returns:
        dict mapping sample index -> per-prefix PPL value
    """
    prefix_length = ppl_config.prefix_length
    sample_size = min(ppl_config.sample_size, len(dataset))
    quick_prompt = ppl_config.quick_prompt

    # Sample indices
    rng = np.random.default_rng()
    sampled_indices = sorted(rng.choice(len(dataset), size=sample_size, replace=False).tolist())

    # Build QAP prompts and tokenize
    all_input_ids = []
    all_attention_mask = []

    for idx in sampled_indices:
        # Get the original chat content from the dataset dataframe
        row = dataset.dataframe.iloc[idx]
        original_chat = row['prompt']

        # Apply QAP
        qap_prompt_str = _build_qap_prompt(original_chat, quick_prompt)

        # Tokenize
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=qap_prompt_str,
            tokenizer=tokenizer,
            max_length=dataset.max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation='error'
        )

        all_input_ids.append(input_ids[0])
        all_attention_mask.append(attention_mask[0])

    # Stack into tensors
    input_ids = torch.stack(all_input_ids)
    attention_mask = torch.stack(all_attention_mask)
    position_ids = compute_position_id_with_mask(attention_mask)

    # Process in chunks to avoid OOM
    entropy_dict = {}
    chunk_size = 64

    for start in range(0, len(sampled_indices), chunk_size):
        end = min(start + chunk_size, len(sampled_indices))
        batch_indices = sampled_indices[start:end]

        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]
        batch_position_ids = position_ids[start:end]

        # Create DataProto for generation
        gen_batch = DataProto.from_dict({
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'position_ids': batch_position_ids,
        })
        gen_batch.meta_info = {
            'eos_token_id': tokenizer.eos_token_id,
            'pad_token_id': tokenizer.pad_token_id,
            'do_sample': True,
            'validate': False,
            'max_tokens': prefix_length,  # override: only generate prefix_length tokens
        }

        # Pad to be divisible by dp_size
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            gen_batch, actor_rollout_wg.world_size
        )

        # Generate short prefixes via rollout worker
        gen_output_padded = actor_rollout_wg.generate_sequences(gen_batch_padded)

        # Unpad
        gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size) if pad_size > 0 else gen_output_padded

        # Build input for log-prob computation: need input_ids, attention_mask, position_ids, responses
        log_prob_input = DataProto.from_dict({
            'input_ids': gen_output.batch['input_ids'],
            'attention_mask': gen_output.batch['attention_mask'],
            'position_ids': gen_output.batch['position_ids'],
            'responses': gen_output.batch['responses'],
        })
        log_prob_input.meta_info = {
            'micro_batch_size': 64,
            'max_token_len': 32768,
            'use_dynamic_bsz': True,
            'temperature': 1.0,
        }

        # Pad for dp
        log_prob_padded, pad_size2 = pad_dataproto_to_divisor(
            log_prob_input, actor_rollout_wg.world_size
        )

        # Compute log-probs
        log_prob_output_padded = actor_rollout_wg.compute_log_prob(log_prob_padded)
        log_prob_output = unpad_dataproto(log_prob_output_padded, pad_size=pad_size2) if pad_size2 > 0 else log_prob_output_padded

        # Extract log-probs and compute prefix entropy
        old_log_probs = log_prob_output.batch['old_log_probs']  # (batch_size * n, response_length)
        responses = gen_output.batch['responses']
        response_length = responses.shape[-1]

        # Response mask from attention_mask
        response_mask = gen_output.batch['attention_mask'][:, -response_length:]

        # Use only up to prefix_length tokens
        actual_prefix_len = min(prefix_length, response_length)
        prefix_log_probs = old_log_probs[:, :actual_prefix_len]
        prefix_mask = response_mask[:, :actual_prefix_len]

        # PPL_i = exp(-mean(log P(y_t | y_{<t}, x'))) for valid prefix tokens
        # Per-token normalization avoids bias toward longer prefixes
        valid_token_count = prefix_mask.sum(dim=-1).clamp(min=1)
        entropy_per_sample = -(prefix_log_probs * prefix_mask).sum(dim=-1) / valid_token_count

        # If n > 1, each prompt has n responses. Only use the first response per prompt.
        # The responses are interleaved: [p0_r0, p0_r1, ..., p1_r0, p1_r1, ...]
        # Batch_indices has len = chunk_size, but the output has len = chunk_size * n
        n_repeats = gen_output.batch['input_ids'].shape[0] // len(batch_indices)
        if n_repeats > 1:
            # Take only the first response per prompt (index 0, n, 2n, ...)
            entropy_per_sample = entropy_per_sample[::n_repeats]

        for i, idx in enumerate(batch_indices):
            entropy_dict[int(idx)] = entropy_per_sample[i].item()

    return entropy_dict
