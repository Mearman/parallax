"""A stage that drafts keeps the multi-token-prediction weights and the embedding table whatever layers it holds."""

import json
from pathlib import Path

from parallax.utils.weight_filter_utils import filter_weight_files_by_layer_range_for_load, should_include_weight_key

LAYERS = 64


def _checkpoint(tmp_path: Path) -> tuple[Path, list[str]]:
    weight_map = {"model.language_model.embed_tokens.weight": "model-00001.safetensors"}
    for layer in range(LAYERS):
        file = "model-00001.safetensors" if layer < 20 else "model-00002.safetensors" if layer < 44 else "model-00003.safetensors"
        weight_map[f"model.language_model.layers.{layer}.mlp.up_proj.weight"] = file
    weight_map["model.language_model.norm.weight"] = "model-00003.safetensors"
    weight_map["lm_head.weight"] = "model-00003.safetensors"
    weight_map["mtp.fc.weight"] = "model-00004.safetensors"
    weight_map["mtp.layers.0.mlp.up_proj.weight"] = "model-00004.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    files = sorted(set(weight_map.values()))
    return tmp_path, [str(tmp_path / name) for name in files]


def _kept(tmp_path: Path, start: int, end: int, include_speculative: bool) -> list[str]:
    model_path, files = _checkpoint(tmp_path)
    kept = filter_weight_files_by_layer_range_for_load(
        model_path=model_path,
        weight_files=files,
        start_layer=start,
        end_layer=end,
        is_first_shard=start == 0,
        is_last_shard=end == LAYERS,
        config={"tie_word_embeddings": False},
        include_speculative=include_speculative,
    )
    return [Path(f).name for f in kept]


def test_a_last_stage_without_speculation_skips_the_files_it_does_not_need():
    assert _kept(Path(__import__("tempfile").mkdtemp()), 44, LAYERS, False) == ["model-00003.safetensors"]


def test_a_last_stage_that_drafts_also_keeps_the_embedding_and_multi_token_prediction_files():
    kept = _kept(Path(__import__("tempfile").mkdtemp()), 44, LAYERS, True)
    assert "model-00001.safetensors" in kept, "the drafter's own embedding table"
    assert "model-00004.safetensors" in kept, "the multi-token-prediction layer"
    assert "model-00002.safetensors" not in kept, "layers the stage does not hold"


def test_the_key_filter_names_what_a_drafting_stage_adds():
    for key in ("mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight", "model.embed_tokens.weight"):
        assert not should_include_weight_key(key, 44, LAYERS, False, True), key
        assert should_include_weight_key(key, 44, LAYERS, False, True, include_speculative=True), key
    assert not should_include_weight_key("model.layers.2.mlp.up_proj.weight", 44, LAYERS, False, True, include_speculative=True)
