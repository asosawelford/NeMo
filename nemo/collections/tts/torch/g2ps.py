# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import pathlib
import random
import re
import time
from collections import defaultdict
from typing import Optional

import nltk
import torch

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.utils import logging
from nemo.utils.decorators import experimental
from nemo.utils.get_rank import is_global_rank_zero
""" from https://github.com/keithito/tacotron """

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

# https://en.wikipedia.org/wiki/ARPABET
valid_symbols = [
  'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
  'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
  'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
  'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
  'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
  'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
  'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', "RR", "NI"
]

# IPA to ARPABET

replacements = {
  'aɪ': 'AA0 Y',
  'aʊ': 'AA0 W',
  'b': 'B',
  'd': 'D',
  'dʒ': 'JH',
  'eɪ': 'EY0',
  'f': 'F',
  'h': 'HH',
  'i': 'IY0',
  'j': 'Y',
  'k': 'K',
  'l': 'L',
  'l̩': 'EL',
  'm': 'M',
  'm̩': 'EM',
  'n': 'N',
  'n̩': 'EN',
  'oʊ': 'OW',
  'p': 'P',
  's': 'S',
  't': 'T',
  'tʃ': 'CH',
  'u': 'UW0',
  'v': 'V',
  'w': 'W',
  'z': 'Z',
  'æ': 'AE0',
  'ð': 'DH',
  'ŋ': 'NG',
  'ɑ': 'AA0',
  'ɔ': 'AO0',
  'ɔɪ': 'OY0',
  'ə': 'AX',
  'ɚ': 'AXR',
  'ɛ': 'EH0',
  'ɝ': 'ER',
  'ɡ': 'G',
  'ɨ': 'IX',
  'ɪ': 'IH0',
  'ɹ': 'R',
  'ɾ': 'R',
  'ɾ̃': 'NX',
  'ʃ': 'SH',
  'ʉ': 'UX',
  'ʊ': 'UH0',
  'ʌ': 'AH0',
  'ʍ': 'WH',
  'ʒ': 'ZH',
  'ʔ': 'Q',
  'θ': 'TH'
}

spanish_replacements = {
    "a": "AA0",
    "o": "OW0",
    "β": "V",
    "e": "EY0",
    "ɲ": "NI",
    "r": "RR",
    "ɣ": "G",
    "ʎ": "SH",
    "x": "HH",
    "ts": "TH", # jacuzzi / pizza
    "oɪ": "OW0 Y", # acoitar
    "eʊ": "EY0 W", # adeuda
    "ʝ": "SH" # ya
}

replacements = {**replacements, **spanish_replacements}

_valid_symbol_set = set(valid_symbols)


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, lang="es"):
    self.separator = Separator(phone='|', word=None)
    self.backend = EspeakBackend(lang,with_stress=True, preserve_punctuation=False)

  def grapheme2ipa(self, word):
    return self.backend.phonemize([word],separator=self.separator)[0].strip().strip("|")

  def ipa2arpabet(self, ipa_word):
    #print(ipa_word)
    ipa_word = ipa_word.replace("ɾ|ɾ", "r")
    if ipa_word.endswith("j|j"):
      ipa_word = ipa_word.replace("j|j", "i")
    else:
      ipa_word = ipa_word.replace("j|j", "ʎ")
    ipa_word = ipa_word.replace("ˌ", "") # Remove secondary accents, they are not important in spanish
    ipa_word = ipa_word.replace("ː", "") # Remove long consonant indication
    arpabet_word = ""
    for c in ipa_word.split("|"):
      accent = False
      if c.startswith("ˈ"):
        c = c[1:]
        accent = True
      if c in replacements:
        replacement = replacements[c]
        if accent:
          replacement = replacement.replace("0","1")
        arpabet_word += replacement + " "
      else:
        print(ipa_word)
        print(f"'{c}' no esta en la lista de replacements")
        raise Exception()
    return arpabet_word.strip().split(" ")

  def word2phonemes(self, word):
    return self.ipa2arpabet(self.grapheme2ipa(word))

  def __len__(self):
    return len(self._entries)

  def lookup(self, word):
    '''Converts word to spainsh adapted ARPAbet'''
    return self.word2phonemes(word)

class BaseG2p(abc.ABC):
    def __init__(
        self, phoneme_dict=None, word_tokenize_func=lambda x: x, apply_to_oov_word=None,
    ):
        """Abstract class for creating an arbitrary module to convert grapheme words to phoneme sequences (or leave unchanged or use apply_to_oov_word).
        Args:
            phoneme_dict: Arbitrary representation of dictionary (phoneme -> grapheme) for known words.
            word_tokenize_func: Function for tokenizing text to words.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
        """
        self.phoneme_dict = phoneme_dict
        self.word_tokenize_func = word_tokenize_func
        self.apply_to_oov_word = apply_to_oov_word

    @abc.abstractmethod
    def __call__(self, text: str) -> str:
        pass


class EnglishG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        word_tokenize_func=english_word_tokenize,
        apply_to_oov_word=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='latin-1',
        phoneme_probability: Optional[float] = None,
    ):
        """English G2P module. This module converts words from grapheme to phoneme representation using phoneme_dict in CMU dict format.
        Optionally, it can ignore words which are heteronyms, ambiguous or marked as unchangeable by word_tokenize_func (see code for details).
        Ignored words are left unchanged or passed through apply_to_oov_word for handling.
        Args:
            phoneme_dict (str, Path, Dict): Path to file in CMUdict format or dictionary of CMUdict-like entries.
            word_tokenize_func: Function for tokenizing text to words.
                It has to return List[Tuple[Union[str, List[str]], bool]] where every tuple denotes word representation and flag whether to leave unchanged or not.
                It is expected that unchangeable word representation will be represented as List[str], other cases are represented as str.
                It is useful to mark word as unchangeable which is already in phoneme representation.
            apply_to_oov_word: Function that will be applied to out of phoneme_dict word.
            ignore_ambiguous_words: Whether to not handle word via phoneme_dict with ambiguous phoneme sequences. Defaults to True.
            heteronyms (str, Path, List): Path to file with heteronyms (every line is new word) or list of words.
            encoding: Encoding type.
            phoneme_probability (Optional[float]): The probability (0.<var<1.) that each word is phonemized. Defaults to None which is the same as 1.
                Note that this code path is only run if the word can be phonemized. For example: If the word does not have a entry in the g2p dict, it will be returned
                as characters. If the word has multiple entries and ignore_ambiguous_words is True, it will be returned as characters.
        """
        phoneme_dict = (
            self._parse_as_cmu_dict(phoneme_dict, encoding)
            if isinstance(phoneme_dict, str) or isinstance(phoneme_dict, pathlib.Path) or phoneme_dict is None
            else phoneme_dict
        )

        if apply_to_oov_word is None:
            logging.warning(
                "apply_to_oov_word=None, This means that some of words will remain unchanged "
                "if they are not handled by any of the rules in self.parse_one_word(). "
                "This may be intended if phonemes and chars are both valid inputs, otherwise, "
                "you may see unexpected deletions in your input."
            )

        super().__init__(
            phoneme_dict=phoneme_dict, word_tokenize_func=word_tokenize_func, apply_to_oov_word=apply_to_oov_word,
        )

        self.ignore_ambiguous_words = ignore_ambiguous_words
        self.heteronyms = (
            set(self._parse_file_by_lines(heteronyms, encoding))
            if isinstance(heteronyms, str) or isinstance(heteronyms, pathlib.Path)
            else heteronyms
        )
        self.phoneme_probability = phoneme_probability
        self._rng = random.Random()
      
        self.espeak = CMUDict(lang="es")
    
    @staticmethod
    def _parse_as_cmu_dict(phoneme_dict_path=None, encoding='latin-1'):
        if phoneme_dict_path is None:
            # this part of code downloads file, but it is not rank zero guarded
            # Try to check if torch distributed is available, if not get global rank zero to download corpora and make
            # all other ranks sleep for a minute
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = torch.distributed.group.WORLD
                if is_global_rank_zero():
                    try:
                        nltk.data.find('corpora/cmudict.zip')
                    except LookupError:
                        nltk.download('cmudict', quiet=True)
                torch.distributed.barrier(group=group)
            elif is_global_rank_zero():
                logging.error(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. Now downloading corpora from global rank 0. If other ranks pass this "
                    "before rank 0, errors might result."
                )
                try:
                    nltk.data.find('corpora/cmudict.zip')
                except LookupError:
                    nltk.download('cmudict', quiet=True)
            else:
                logging.error(
                    f"Torch distributed needs to be initialized before you initialized EnglishG2p. This class is prone to "
                    "data access race conditions. This process is not rank 0, and now going to sleep for 1 min. If this "
                    "rank wakes from sleep prior to rank 0 finishing downloading, errors might result."
                )
                time.sleep(60)

            logging.warning(
                f"English g2p_dict will be used from nltk.corpus.cmudict.dict(), because phoneme_dict_path=None. "
                "Note that nltk.corpus.cmudict.dict() has old version (0.6) of CMUDict. "
                "You can use the latest official version of CMUDict (0.7b) with additional changes from NVIDIA directly from NeMo "
                "using the path scripts/tts_dataset_files/cmudict-0.7b_nv22.08."
            )

            return nltk.corpus.cmudict.dict()

        _alt_re = re.compile(r'\([0-9]+\)')
        g2p_dict = {}
        with open(phoneme_dict_path, encoding=encoding) as file:
            for line in file:
                if len(line) and ('A' <= line[0] <= 'Z' or line[0] == "'"):
                    parts = line.split('  ')
                    word = re.sub(_alt_re, '', parts[0])
                    word = word.lower()

                    pronunciation = parts[1].strip().split(" ")
                    if word in g2p_dict:
                        g2p_dict[word].append(pronunciation)
                    else:
                        g2p_dict[word] = [pronunciation]
        return g2p_dict

    @staticmethod
    def _parse_file_by_lines(p, encoding):
        with open(p, encoding=encoding) as f:
            return [l.rstrip() for l in f.readlines()]

    def is_unique_in_phoneme_dict(self, word):
        return len(self.phoneme_dict[word]) == 1

    def parse_one_word(self, word: str):
        """
        Returns parsed `word` and `status` as bool.
        `status` will be `False` if word wasn't handled, `True` otherwise.
        """

        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return word, True

        # Punctuation (assumes other chars have been stripped)
        if re.search(r"[A-Za-zÀ-ÿ]", word) is None:
            return list(word), True

        return self.espeak.lookup(word), True
   
    def __call__(self, text):
        words = self.word_tokenize_func(text)

        prons = []
        for word, without_changes in words:
            if without_changes:
                prons.extend(word)
                continue

            word_by_hyphen = word.split("-")

            pron, is_handled = self.parse_one_word(word)

            if not is_handled and len(word_by_hyphen) > 1:
                pron = []
                for sub_word in word_by_hyphen:
                    p, _ = self.parse_one_word(sub_word)
                    pron.extend(p)
                    pron.extend(["-"])
                pron.pop()

            prons.extend(pron)

        return prons
   
