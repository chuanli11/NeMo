import pytest

import nemo.lightning as nl
from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

DATA_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/simple_wiki_gpt_preproc_text_document"
VOCAB_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/vocab.json"
MERGES_PATH = "/home/TestData/nlp/megatron_gpt/data/gpt/merges.txt"


@pytest.fixture
def tokenizer():
    return get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=VOCAB_PATH,
        merges_file=MERGES_PATH,
    )


@pytest.fixture
def trainer():
    return nl.Trainer(
        accelerator="cpu",
        max_steps=1,
    )


def test_single_data_distribution(tokenizer, trainer):

    data = PreTrainingDataModule(
        paths=[DATA_PATH],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
        tokenizer=tokenizer,
    )
    data.trainer = trainer

    ## AssertioneError because we are trying to do eval on the whole
    ## dataset with just a single distribution
    with pytest.raises(AssertionError):
        data.setup(stage="dummy")

    trainer.limit_val_batches = 5
    ## this should succeed
    data.setup(stage="dummy")


def test_multiple_data_distributions(tokenizer, trainer):
    data = PreTrainingDataModule(
        paths={
            "train": ['1', DATA_PATH],
            "validation": [DATA_PATH, DATA_PATH],
            "test": ['1', DATA_PATH],
        },
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
        tokenizer=tokenizer,
    )
    data.trainer = trainer

    ## this should succeed
    data.setup(stage="dummy")


def test_validate_dataset_asset_accessibility_file_does_not_exist(tokenizer, trainer):
    raised_exception = False
    try:
        data = PreTrainingDataModule(
            paths=["/this/path/should/not/exist/"],
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
            tokenizer=tokenizer,
        )
        data.trainer = trainer
    except FileNotFoundError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a FileNotFoundError"


def test_validate_dataset_asset_accessibility_file_is_none(tokenizer, trainer):
    raised_exception = False
    try:
        data = PreTrainingDataModule(
            paths=None,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
            tokenizer=tokenizer,
        )
        data.trainer = trainer
    except ValueError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a ValueError"
