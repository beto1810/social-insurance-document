import yaml
from pathlib import Path


class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_name(name):
        vietocr_root = Path(__file__).resolve().parents[2]

        candidates = [
            vietocr_root / "config",
            vietocr_root / "vietocr" / "config",
        ]

        config_dir = next(
            (p for p in candidates if (p / "base.yml").exists()),
            None,
        )

        if config_dir is None:
            raise FileNotFoundError(
                "Could not find VietOCR config directory. "
                f"Checked: {candidates}"
            )

        model_files = {
            "vgg_transformer": "vgg-transformer.yml",
            "resnet_transformer": "resnet_transformer.yml",
            "resnet_fpn_transformer": "resnet_fpn_transformer.yml",
            "vgg_seq2seq": "vgg-seq2seq.yml",
            "vgg_convseq2seq": "vgg_convseq2seq.yml",
            "vgg_decoderseq2seq": "vgg_decoderseq2seq.yml",
        }

        if name not in model_files:
            raise ValueError(f"Unsupported config: {name}")

        base_file = config_dir / "base.yml"
        model_file = config_dir / model_files[name]

        with open(base_file, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        with open(model_file, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)

        base_config.update(model_config)

        return Cfg(base_config)

    def save(self, fname):
        with open(fname, "w", encoding="utf-8") as outfile:
            yaml.dump(
                dict(self),
                outfile,
                default_flow_style=False,
                allow_unicode=True,
            )