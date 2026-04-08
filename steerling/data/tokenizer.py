"""
Steerling tokenizer: tiktoken cl100k_base with custom special tokens.

Token layout:
    0 - 100256:  cl100k_base base vocabulary
    100257:      <|endoftext|> (EOS, built into tiktoken)
    100277:      <|pad|>
    100278:      <|bos|>
    100279:      <|endofchunk|>
    100280:      <|mask|>
    100281:      <|im_start|> (ChatML turn start)
    100282:      <|im_end|> (ChatML turn end)
    vocab_size:  100283
"""

from __future__ import annotations

import numpy as np
import tiktoken
import torch


class SteerlingTokenizer:
    """
    Tokenizer for Steerling models.

    Uses tiktoken cl100k_base with 4 additional special tokens as mentioned above.
    """

    ENCODING_NAME = "cl100k_base"

    def __init__(self):
        base_enc = tiktoken.get_encoding(self.ENCODING_NAME)
        base_vocab = base_enc.n_vocab  # 100277

        self._pad_token_id = base_vocab  # 100277
        self._bos_token_id = base_vocab + 1  # 100278
        self._endofchunk_token_id = base_vocab + 2  # 100279
        self._mask_token_id = base_vocab + 3  # 100280
        self._eos_token_id = base_enc._special_tokens["<|endoftext|>"]  # 100257
        self._im_start_token_id = base_vocab + 4  # 100281
        self._im_end_token_id = base_vocab + 5  # 100282
        self._vocab_size = base_vocab + 6  # 100283

        # Create encoding with custom special tokens
        self._tokenizer = tiktoken.Encoding(
            name=f"{self.ENCODING_NAME}_steerling",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={
                **base_enc._special_tokens,
                "<|pad|>": self._pad_token_id,
                "<|bos|>": self._bos_token_id,
                "<|endofchunk|>": self._endofchunk_token_id,
                "<|mask|>": self._mask_token_id,
                "<|im_start|>": self._im_start_token_id,
                "<|im_end|>": self._im_end_token_id,
            },
        )

        self._special_token_ids = {
            self._pad_token_id,
            self._bos_token_id,
            self._eos_token_id,
            self._endofchunk_token_id,
            self._mask_token_id,
            self._im_start_token_id,
            self._im_end_token_id,
        }

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: If True, prepend BOS and append EOS

        Returns:
            List of token IDs
        """

        tokens = self._tokenizer.encode(text, allowed_special="all")
        if add_special_tokens:
            tokens = [self._bos_token_id] + tokens + [self._eos_token_id]
        return tokens

    def decode(self, tokens: list[int] | np.ndarray | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: Token IDs (list, numpy array, or torch tensor)
            skip_special_tokens: If True, filter out special tokens before decoding

        Returns:
            Decoded text
        """

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if skip_special_tokens:
            tokens = [int(t) for t in tokens if int(t) not in self._special_token_ids]
        else:
            tokens = [int(t) for t in tokens]
        return self._tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def endofchunk_token_id(self) -> int:
        return self._endofchunk_token_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def im_start_token_id(self) -> int:
        return self._im_start_token_id

    @property
    def im_end_token_id(self) -> int:
        return self._im_end_token_id

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format messages as a ChatML string.

        Args:
            messages: List of dicts with "role" and "content" keys.
                      Supported roles: "system", "user", "assistant".
            add_generation_prompt: If True, append the assistant turn header
                                  (for inference). If False, close the last
                                  assistant turn with <|im_end|> (for training).

        Returns:
            ChatML-formatted string.
        """
        parts: list[str] = []
        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            is_last = i == len(messages) - 1

            parts.append(f"<|im_start|>{role}\n{content}")

            if is_last and role == "assistant" and add_generation_prompt:
                # During inference: leave the last assistant turn open
                # (caller already provided a partial assistant message)
                parts.append("<|im_end|>")
            else:
                parts.append("<|im_end|>\n")

        if add_generation_prompt:
            last_role = messages[-1]["role"] if messages else None
            if last_role != "assistant":
                parts.append("<|im_start|>assistant\n")

        return "".join(parts)
