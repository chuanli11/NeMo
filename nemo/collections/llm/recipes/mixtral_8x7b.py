from typing import Optional

import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from pytorch_lightning.callbacks.callback import Callback

from nemo import lightning as nl
from nemo.collections.llm.api import finetune, pretrain
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.gpt.model.mixtral import MixtralConfig8x7B, MixtralModel
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.recipes.log.default import default_log, default_resume, tensorboard_logger
from nemo.collections.llm.recipes.optim.adam import distributed_fused_adam_with_cosine_annealing
from nemo.utils.exp_manager import TimingCallback

NAME = "mixtral_8x7b"


@run.cli.factory(name=NAME)
def model() -> run.Config[pl.LightningModule]:
    return run.Config(MixtralModel, config=run.Config(MixtralConfig8x7B))


def trainer(
    tensor_parallelism: int,
    pipeline_parallelism: int,
    pipeline_parallelism_type: Optional[torch.dtype],
    virtual_pipeline_parallelism: Optional[int],
    context_parallelism: int,
    sequence_parallelism: bool,
    expert_parallelism: int,
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
    max_steps: int = 1168251,
    callbacks: Optional[list[run.Config[Callback]]] = None,
) -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_type,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_model_parallel_size=expert_parallelism,
        gradient_as_bucket_view=True,
        ckpt_include_optimizer=True,
        ckpt_async_save=True,
        ckpt_parallel_load=True,
        ddp=run.Config(
            DistributedDataParallelConfig,
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
        ),
    )

    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        accumulate_grad_batches=1,
        callbacks=callbacks,
        devices=num_gpus_per_node,
        gradient_clip_val=1.0,
        limit_test_batches=50,
        limit_val_batches=32,
        log_every_n_steps=10,
        max_steps=max_steps,
        num_nodes=num_nodes,
        plugins=run.Config(nl.MegatronMixedPrecision, precision="bf16-mixed"),
        strategy=strategy,
        use_distributed_sampler=False,
        val_check_interval=2000,
    )

    return trainer


@run.cli.factory(target=pretrain, name=NAME)
def pretrain_recipe(
    dir: Optional[str] = None, name: str = "default", num_nodes: int = 1, num_gpus_per_node: int = 8, fn=pretrain
) -> run.Partial:
    return run.Partial(
        fn,
        model=model(),
        trainer=trainer(
            tensor_parallelism=8,
            pipeline_parallelism=1,
            pipeline_parallelism_type=None,
            virtual_pipeline_parallelism=None,
            context_parallelism=1,
            sequence_parallelism=True,
            expert_parallelism=1,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            callbacks=[run.Config(TimingCallback)],
        ),
        data=run.Config(MockDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1),
        log=default_log(dir=dir, name=name, tensorboard_logger=tensorboard_logger(name=name)),
        optim=distributed_fused_adam_with_cosine_annealing(max_lr=3e-4),
        resume=default_resume(),
    )


def hf_resume() -> run.Config[nl.AutoResume]:
    return run.Config(nl.AutoResume, import_path="hf://mistralai/Mixtral-8x7B-v0.1")


@run.cli.factory(target=finetune, name=NAME)
def finetune_recipe(
    dir: Optional[str] = None,
    name: str = "default",
    num_nodes: int = 1,
    num_gpus_per_node: int = 8,
) -> run.Partial:
    recipe = pretrain_recipe(name=name, dir=dir, num_nodes=num_nodes, num_gpus_per_node=num_gpus_per_node, fn=finetune)
    recipe.resume = hf_resume()
    recipe.peft = run.Config(LoRA, target_modules=['linear_qkv', 'linear_proj'], dim=32)
    recipe.data = run.Config(SquadDataModule, seq_length=8192, global_batch_size=512, micro_batch_size=1)
    return recipe
