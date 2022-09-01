from tokenizers import (
        Tokenizer,
        AddedToken,
        pre_tokenizers,
        processors,
        decoders,
        trainers,
        Regex,
        NormalizedString,
        PreTokenizedString
    )
from tokenizers.models import WordPiece, BPE
from tokenizers.normalizers import Replace, Lowercase, Sequence
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from typing import Optional, Tuple, List, Union, Dict, Iterator

class KmerPreTokenizer():
    """
    KmerLevel PreTokenizer
    Args:
    """

    def __init__(self, k):
        self.k = k
        pass
    @staticmethod
    def alphabet():
        """
        Returns the alphabet used by this PreTokenizer.
        Since the ByteLevel works as its name suggests, at the byte level, it
        """
        return list('ACTG')
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place
        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`
        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pretok.split(self._kmer_split)
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string
        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`
        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize
        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        return [(str(k), (i, i+self.k)) for i, k in enumerate(self._kmer_split(0, NormalizedString(sequence)))]
    def _inter_space(self, i, sequence: NormalizedString) -> NormalizedString:
        if i != 0:
            sequence.prepend(" ")
        return sequence
    def _kmer_split(self, i, sequence: NormalizedString) -> List[NormalizedString]:
        return [self._inter_space(j, sequence[j: j + self.k]) for j in range(len(str(sequence)) - self.k + 1)]

class KmerTokenizer(BaseTokenizer):
    """ Kmer WordPiece Tokenizer """

    def __init__(
        self,
        k,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        unknow_nucleotide: bool = True,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):
        self.k = k
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(unk_token)))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(unk_token)))

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        normalizers = []

        if unknow_nucleotide:
            normalizers += [Replace(r'[^ACTG]', '')]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(KmerPreTokenizer(5))

        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = processors.BertProcessing(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
            )
        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        tokenizer.decoder = decoders.WordPiece()

        parameters = {
            "model": "BertWordPiece",
            "k": k,
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "unknow_nucleotide": unknow_nucleotide,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordPiece.read_file(vocab)
        return KmerTokenizer(vocab, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
        length: Optional[int] = None,
    ):
        """ Train the model using the given iterator """

        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

class KmerBPETokenizer(BaseTokenizer):
    """ByteLevelBPETokenizer

    Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    """

    def __init__(
        self,
        k,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        add_prefix_space: bool = False,
        unk_token: Union[str, AddedToken] = "[UNK]",
        sep_token: Union[str, AddedToken] = "[SEP]",
        cls_token: Union[str, AddedToken] = "[CLS]",
        pad_token: Union[str, AddedToken] = "[PAD]",
        mask_token: Union[str, AddedToken] = "[MASK]",
        unknow_nucleotide: bool = True,
        lowercase: bool = True,
        trim_offsets: bool = False,
        max_length: int = 512,
        **kwargs
    ):
        self.k =k
        if vocab is not None and merges is not None:
            tokenizer = Tokenizer(
                BPE(
                    vocab,
                    merges,
                    unk_token=unk_token,
                    **kwargs
                )
            )
        else:
            tokenizer = Tokenizer(BPE())

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is None:
            tokenizer.add_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is None:
            tokenizer.add_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is None:
            tokenizer.add_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is None:
            tokenizer.add_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is None:
            tokenizer.add_tokens([str(mask_token)])

        normalizers = []

        if unknow_nucleotide:
            normalizers += [Replace(r'[^ACTG]', '')]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.PreTokenizer.custom(KmerPreTokenizer(self.k)),
            pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=trim_offsets)

        parameters = {
            "model": "ByteLevelBPE",
            "k": k,
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "add_prefix_space": add_prefix_space,
            "unknow_nucleotide": unknow_nucleotide,
            "lowercase": lowercase,
            "trim_offsets": trim_offsets
        }

        super().__init__(tokenizer, parameters, **kwargs)

    def enable_truncation(self, **kwargs):
        return self._tokenizer.enable_truncation(**kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
        ],
        length: Optional[int] = None,
    ):
        """ Train the model using the given iterator """

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )
