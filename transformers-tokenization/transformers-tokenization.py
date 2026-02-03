import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        # 1️⃣ Add special tokens first
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        for token in special_tokens:
            self.word_to_id[token] = self.vocab_size
            self.id_to_word[self.vocab_size] = token
            self.vocab_size += 1

        # 2️⃣ Collect unique words from texts
        unique_words = set()
        for text in texts:
            for word in text.split():
                unique_words.add(word)

        # 3️⃣ Add words to vocab
        for word in sorted(unique_words):
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        encoded = []
        for word in text.split():
            encoded.append(
                self.word_to_id.get(word, self.word_to_id[self.unk_token])
            )
        return encoded

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        words = []
        for i in ids:
            words.append(self.id_to_word.get(i, self.unk_token))
        return " ".join(words)
