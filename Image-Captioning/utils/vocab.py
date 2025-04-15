from typing import List
import numpy as np
from numpy.typing import ArrayLike

class GenerationVocab:
    """
    GenerationVocab is a utility class for managing a vocabulary of tokens used in 
    natural language processing tasks. It provides methods for tokenizing, padding, 
    and detokenizing sequences, as well as preparing and unpreparing data for model 
    input and output.
    Attributes:
        START_TAG (str): Special token indicating the start of a sequence.
        STOP_TAG (str): Special token indicating the end of a sequence.
        PAD_TAG (str): Special token used for padding sequences to a fixed size.
        UNKNOWN_TAG (str): Special token used for unknown or out-of-vocabulary tokens.
        AUX_TAGS (List[str]): List of auxiliary tokens including START_TAG, STOP_TAG, 
            PAD_TAG, and UNKNOWN_TAG.
    Methods:
        __init__(VOCAB: List):
            Initializes the vocabulary with auxiliary tokens and a custom list of tokens.
        __len__() -> int:
            Returns the total number of tokens in the vocabulary.
        tokenise(line: List[str]) -> List[int]:
            Converts a list of string tokens into their corresponding integer indices 
            based on the vocabulary. Unknown tokens are replaced with the UNKNOWN_TAG index.
        pad(tokenised: List[int], size: int) -> ArrayLike:
            Pads a tokenized sequence to a fixed size, adding START_TAG at the beginning, 
            STOP_TAG at the end, and PAD_TAG in between as needed.
        prepare(line: List[str], size: int) -> ArrayLike:
            Tokenizes and pads a sequence to prepare it for model input.
        unpad(padded: ArrayLike) -> List[int]:
            Removes padding and special tokens (START_TAG, PAD_TAG, STOP_TAG) from a 
            padded sequence, returning the original token indices.
        detokenise(tokenised: List[int]) -> List[str]:
            Converts a list of integer token indices back into their corresponding string 
            tokens based on the vocabulary.
        unprepare(padded: ArrayLike) -> List[str]:
            Reverses the preparation process by unpadding and detokenizing a padded 
            sequence, returning the original list of string tokens.
    """
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    UNKNOWN_TAG = "<UNK>"

    AUX_TAGS = [START_TAG, STOP_TAG, PAD_TAG, UNKNOWN_TAG]

    def __init__(self, VOCAB:List) -> None:
        self.tokens = self.AUX_TAGS + VOCAB

        self.token2index = {tok: ii for ii, tok in enumerate(self.tokens)}
        self.index2token = {ii: tok for tok, ii in self.token2index.items()}

    def __len__(self) -> int:
        return len(self.token2index)

    def tokenise(self, line: List[str]) -> List[int]:
        
        return [
            self.token2index[tok]
            if tok in self.token2index
            else self.token2index[self.UNKNOWN_TAG]
            for tok in line
        ]

    def pad(self, tokenised: List[int], size: int) -> ArrayLike:
        padded = np.full((size,), self.token2index[self.PAD_TAG])
        max_index = min(len(tokenised), size - 2)
        padded[1 : max_index + 1] = tokenised[:max_index]
        padded[0] = self.token2index[self.START_TAG]
        padded[max_index + 1] = self.token2index[self.STOP_TAG]
        return padded

    def prepare(self, line: List[str], size: int) -> ArrayLike:
        return self.pad(self.tokenise(line), size)

    def unpad(self, padded: ArrayLike) -> List[int]:
        output: List[int] = []
        for tok in padded:
            if tok not in {
                self.token2index[self.START_TAG],
                self.token2index[self.PAD_TAG],
            }:
                if tok == self.token2index[self.STOP_TAG]:
                    return output
                output.append(tok)

        return output

    def detokenise(self, tokenised: List[int]) -> List[str]:
        return [self.index2token.get(ind, self.UNKNOWN_TAG) for ind in tokenised]

    def unprepare(self, padded: ArrayLike) -> List[str]:
        return self.detokenise(self.unpad(padded))