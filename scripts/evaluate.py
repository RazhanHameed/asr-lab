#!/usr/bin/env python3
"""Evaluation script for ASR Lab models.

Supports evaluation on FLEURS and LibriSpeech datasets.

Usage:
    # Evaluate on FLEURS
    python scripts/evaluate.py --checkpoint outputs/best.pt --dataset fleurs --language en_us

    # Evaluate on LibriSpeech
    python scripts/evaluate.py --checkpoint outputs/best.pt --dataset librispeech --split test.clean

    # Compare streaming vs offline
    python scripts/evaluate.py --checkpoint outputs/best.pt --compare-streaming
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from asr_lab.models.ssm import SSMASRModel
from asr_lab.models.whisper import WhisperASRModel
from asr_lab.models.conformer import ConformerASRModel
from asr_lab.models.base import ModelType
from asr_lab.audio.features import MelSpectrogramExtractor, FeatureConfig
from asr_lab.tokenizers.base import CharacterTokenizer
from asr_lab.training.dataset import ASRDataset, ASRCollator, ASRSample
from asr_lab.evaluation.evaluator import Evaluator, EvaluationConfig


def load_model(checkpoint_path: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model_type = ModelType(config.get("model_type", "ssm"))

    if model_type == ModelType.SSM:
        from asr_lab.models.ssm import SSMConfig

        model_config = SSMConfig.from_dict(config)
        model = SSMASRModel(model_config)
    elif model_type == ModelType.WHISPER:
        from asr_lab.models.whisper import WhisperConfig

        model_config = WhisperConfig.from_dict(config)
        model = WhisperASRModel(model_config)
    elif model_type == ModelType.CONFORMER:
        from asr_lab.models.conformer import ConformerConfig

        model_config = ConformerConfig.from_dict(config)
        model = ConformerASRModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_fleurs_dataset(
    language: str,
    split: str,
    feature_extractor: MelSpectrogramExtractor,
    tokenizer: CharacterTokenizer,
) -> ASRDataset:
    """Load FLEURS dataset for a specific language."""
    try:
        from datasets import load_dataset

        ds = load_dataset("google/fleurs", language, split=split)

        samples = []
        for item in ds:
            samples.append(
                ASRSample(
                    audio_path=item["audio"]["path"],
                    text=item["transcription"].lower(),
                    duration=len(item["audio"]["array"])
                    / item["audio"]["sampling_rate"],
                    language=language,
                )
            )

        return ASRDataset(samples, feature_extractor, tokenizer)

    except ImportError:
        raise ImportError(
            "datasets library required for FLEURS. Install with: pip install datasets"
        )


def load_librispeech_dataset(
    split: str,
    feature_extractor: MelSpectrogramExtractor,
    tokenizer: CharacterTokenizer,
) -> ASRDataset:
    """Load LibriSpeech dataset."""
    try:
        from datasets import load_dataset

        ds = load_dataset("librispeech_asr", split=split)

        samples = []
        for item in ds:
            samples.append(
                ASRSample(
                    audio_path=item["audio"]["path"],
                    text=item["text"].lower(),
                    duration=len(item["audio"]["array"])
                    / item["audio"]["sampling_rate"],
                )
            )

        return ASRDataset(samples, feature_extractor, tokenizer)

    except ImportError:
        raise ImportError(
            "datasets library required. Install with: pip install datasets"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR models")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="fleurs",
        choices=["fleurs", "librispeech"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en_us",
        help="FLEURS language code (comma-separated for multiple, 'all' for all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split",
    )

    # Evaluation arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "fp8", "auto"],
        help="Inference precision",
    )
    parser.add_argument(
        "--compare-streaming",
        action="store_true",
        help="Compare streaming vs offline",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create feature extractor
    feature_config = FeatureConfig(sample_rate=16000, n_mels=80)
    feature_extractor = MelSpectrogramExtractor(feature_config)

    # Create tokenizer
    tokenizer = CharacterTokenizer()

    # Create evaluator config
    eval_config = EvaluationConfig(
        batch_size=args.batch_size,
        precision=args.precision,
        compare_streaming=args.compare_streaming,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # Create evaluator
    evaluator = Evaluator(model, tokenizer, eval_config)

    # Get languages to evaluate
    if args.dataset == "fleurs":
        if args.language == "all":
            # All 102 FLEURS languages
            languages = [
                "af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by",
                "bg_bg", "bn_in", "bs_ba", "ca_es", "ceb_ph", "ckb_iq", "cmn_hans_cn",
                "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419",
                "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie",
                "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu",
                "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id",
                "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr",
                "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke",
                "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my",
                "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw",
                "oc_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br",
                "ro_ro", "ru_ru", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so",
                "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th",
                "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn",
                "xh_za", "yo_ng", "yue_hant_hk", "zu_za",
            ]
        else:
            languages = [lang.strip() for lang in args.language.split(",")]
    else:
        languages = [args.split]

    # Evaluate
    all_results = {}

    for lang in languages:
        print(f"\nEvaluating on {lang}...")

        # Load dataset
        if args.dataset == "fleurs":
            dataset = load_fleurs_dataset(
                lang, args.split, feature_extractor, tokenizer
            )
        else:
            dataset = load_librispeech_dataset(lang, feature_extractor, tokenizer)

        # Create dataloader
        collator = ASRCollator()
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=4,
        )

        # Evaluate
        if args.compare_streaming:
            results = evaluator.compare_streaming(dataloader)
            all_results[lang] = {
                "offline_wer": results["offline"].wer,
                "offline_cer": results["offline"].cer,
                "streaming_wer": results["streaming"].wer,
                "streaming_cer": results["streaming"].cer,
            }
        else:
            result = evaluator.evaluate(dataloader)
            all_results[lang] = {
                "wer": result.wer,
                "cer": result.cer,
                "samples": result.samples,
            }
            print(f"  WER: {result.wer:.2%}")
            print(f"  CER: {result.cer:.2%}")
            print(f"  Samples: {result.samples}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    if len(languages) > 1:
        avg_wer = sum(r.get("wer", r.get("offline_wer", 0)) for r in all_results.values()) / len(
            all_results
        )
        print(f"\nAverage WER across {len(languages)} languages: {avg_wer:.2%}")


if __name__ == "__main__":
    main()
