from itertools import product


FISHER_ROOT = "/scratch-shared/eterres/fishers/sdxl"


def _concept(name, class_name):
    return {
        "name": name,
        "class_name": class_name,
        "placeholder_token": f"<{name}>",
        "adapter_path": f"/scratch-shared/eterres/SDXL/concepts/adapters/pytorch_lora_weights_{name}.safetensors",
        "dataset_path": f"/scratch-shared/eterres/SDXL/concepts/datasets/{name}",
        "fim_path": f"{FISHER_ROOT}/{name}_oft_lie_fim.safetensors",
        "kfac_path": f"{FISHER_ROOT}/{name}_oft_lie_kfac.safetensors",
        "type": "concept",
    }


def _style(name, image):
    return {
        "name": name,
        "class_name": "style",
        "placeholder_token": "<style>",
        "adapter_path": f"/scratch-shared/eterres/SDXL/styles/adapters/pytorch_lora_weights_{name}.safetensors",
        "dataset_path": f"/scratch-shared/eterres/SDXL/styles/datasets/{image}",
        "fim_path": f"{FISHER_ROOT}/{name}_oft_lie_fim.safetensors",
        "kfac_path": f"{FISHER_ROOT}/{name}_oft_lie_kfac.safetensors",
        "type": "style",
    }


CONCEPT_ADAPTERS = [
    _concept("cat", "cat"),
    _concept("cat2", "cat"),
    _concept("dog", "dog"),
    _concept("dog2", "dog"),
    _concept("dog3", "dog"),
    _concept("dog6", "dog"),
]


STYLE_ADAPTERS = [
    _style("01_01", "image_01_01.jpg"),
    _style("01_02", "image_01_02.jpg"),
    _style("01_03", "image_01_03.jpg"),
    _style("01_07", "image_01_07.jpg"),
    _style("01_08", "image_01_08.jpg"),
    _style("02_03", "image_02_03.jpg"),
    _style("03_04", "image_03_04.jpg"),
    _style("dolina", "dolina.png"),
    _style("etsy", "etsy.jpeg"),
    _style("gondoliers", "gondoliers.jpg"),
    _style("image_scan", "image_scan.jpg"),
    _style("pots", "pots.jpeg"),
]


DIFFUSION_MERGE_PAIRS = [
    {
        "name": f"{concept['name']}__{style['name']}",
        "concept": concept,
        "style": style,
    }
    for concept, style in product(CONCEPT_ADAPTERS, STYLE_ADAPTERS)
]


def get_entry(entry_type, name):
    entries = {"concept": CONCEPT_ADAPTERS, "style": STYLE_ADAPTERS}[entry_type]
    for entry in entries:
        if entry["name"] == name:
            return entry
    raise KeyError(f"Unknown {entry_type}: {name}")


def get_pair(concept_name, style_name):
    concept = get_entry("concept", concept_name)
    style = get_entry("style", style_name)
    return {
        "name": f"{concept_name}__{style_name}",
        "concept": concept,
        "style": style,
    }
