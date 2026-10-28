"""Tiny random Llama + in-memory WordLevel tokenizer with a chat template (no download)."""
from __future__ import annotations

WORDS = ("Record for Ada . Ada lives in Paris Rome Oslo . What city does live ? Answer with one word "
         "Yes No is it true that the pet cat dog").split()
SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<|system|>", "<|user|>", "<|assistant|>"]
TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}<|assistant|> {% endif %}")


def tiny_pair(seed: int = 0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(SPECIAL + sorted(set(WORDS)))}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>", additional_special_tokens=SPECIAL[4:])
    tok.chat_template = TEMPLATE
    cfg = LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg), tok
