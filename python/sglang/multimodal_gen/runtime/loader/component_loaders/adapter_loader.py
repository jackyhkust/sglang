from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads a single safetensors state_dict.
    """

    component_names = ["connectors"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        config = get_diffusers_component_config(component_path=component_model_path)

        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        if config.get("per_modality_projections"):
            # SGL-D's own LTX2TextConnectors reimplementation only has the
            # single shared text_proj_in path (LTX-2.0). LTX-2.3/2.5
            # checkpoints use per-modality (separate video/audio)
            # projections, a different architecture branch this class does
            # not implement. Force a fallback to the native (diffusers)
            # implementation instead of silently loading with
            # mismatched/missing weights.
            raise ValueError(
                "per_modality_projections=True (LTX-2.3+) is not supported by "
                "SGL-D's customized LTX2TextConnectors; falling back to native."
            )

        server_args.model_paths["connectors"] = component_model_path

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        target_device = get_local_torch_device()
        default_dtype = resolve_precision(
            server_args, "connectors", precision_attr="dit_precision"
        )

        with set_default_torch_dtype(default_dtype), skip_init_modules():
            connector_cfg = LTX2ConnectorConfig()
            connector_cfg.update_model_arch(config)
            model = model_cls(connector_cfg).to(
                device=target_device, dtype=default_dtype
            )

        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

        if len(safetensors_list) == 1:
            loaded = safetensors_load_file(safetensors_list[0])
        else:
            import os
            import re

            # Some checkpoints (e.g. LTX-2.5's connectors) redundantly ship a
            # fully-consolidated single file alongside a sharded pair with an
            # index.json. Prefer the consolidated (non-shard-suffixed) file
            # when present; otherwise merge shards via the index weight map.
            shard_pattern = re.compile(r"-\d+-of-\d+\.safetensors$")
            consolidated = [f for f in safetensors_list if not shard_pattern.search(f)]
            if len(consolidated) == 1:
                loaded = safetensors_load_file(consolidated[0])
            else:
                index_candidates = [
                    os.path.join(component_model_path, f)
                    for f in os.listdir(component_model_path)
                    if f.endswith(".safetensors.index.json")
                ]
                if not index_candidates:
                    raise ValueError(
                        f"Found {len(safetensors_list)} safetensors files in "
                        f"{component_model_path} with no unambiguous consolidated "
                        "file and no *.safetensors.index.json to merge shards."
                    )
                import json

                with open(index_candidates[0]) as fh:
                    weight_map = json.load(fh)["weight_map"]
                loaded = {}
                shard_cache = {}
                for param_name, shard_file in weight_map.items():
                    shard_path = os.path.join(component_model_path, shard_file)
                    if shard_path not in shard_cache:
                        shard_cache[shard_path] = safetensors_load_file(shard_path)
                    loaded[param_name] = shard_cache[shard_path][param_name]

        model.load_state_dict(loaded, strict=False)

        return model
