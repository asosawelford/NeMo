# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import os

import torch
from PIL import Image
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.clip.megatron_clip_models import MegatronCLIPModel
from nemo.collections.nlp.parts.nlp_overrides import (
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@hydra_runner(config_path="conf", config_name="megatron_clip_infer")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        find_unused_parameters=False,
    )
    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    # trainer required for restoring model parallel models
    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model_cfg = MegatronCLIPModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        return_config=True,
    )

    assert (
            cfg.trainer.devices * cfg.trainer.num_nodes
            == model_cfg.tensor_model_parallel_size * model_cfg.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    # These configs are required to be off during inference.
    with open_dict(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.vision.precision = cfg.trainer.precision
        model_cfg.text.precision = cfg.trainer.precision
        if trainer.precision != "bf16":
            model_cfg.megatron_amp_O2 = False
        model_cfg.sequence_parallel = False
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.activations_checkpoint_method = None

    model = MegatronCLIPModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=save_restore_connector,
        strict=True,
    )
    model.eval().requires_grad_(False)

    if model_cfg.get("megatron_amp_O2", False):
        vision_encoder = model.model.module.vision_encoder
        text_encoder = model.model.module.text_encoder
    else:
        vision_encoder = model.model.vision_encoder
        text_encoder = model.model.text_encoder

    # initialize apex DDP strategy
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    _, val_image_transform, text_transform = get_preprocess_fns(
        model_cfg,
        model.tokenizer,
    )

    # get autocast_dtype
    if trainer.precision == 'bf16':
        autocast_dtype = torch.bfloat16
    elif int(trainer.precision) == 32:
        autocast_dtype = torch.float
    elif int(trainer.precision) == 16:
        autocast_dtype = torch.half
    else:
        raise ValueError('precision must be in [32, 16, "bf16"]')

    image = Image.open(cfg.image_path).convert('RGB')
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_dtype in (torch.half, torch.bfloat16),
                                                  dtype=autocast_dtype, ):
        image = val_image_transform(image).unsqueeze(0).cuda()
        texts = text_transform(cfg.texts).cuda()
        image_features = vision_encoder(image)
        text_features = text_encoder(texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    if is_global_rank_zero:
        print(f"Given image's CLIP text probability: ", list(zip(cfg.texts, text_probs[0].cpu().numpy())))


if __name__ == '__main__':
    main()