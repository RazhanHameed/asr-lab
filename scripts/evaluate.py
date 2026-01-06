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


def load_model(checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Handle nested config structure (model config under 'model' key)
    if "model" in config and isinstance(config["model"], dict):
        model_config_dict = config["model"].copy()
        model_type_str = model_config_dict.pop("type", "ssm")
    else:
        model_config_dict = config.copy()
        model_type_str = model_config_dict.pop("model_type", "ssm")

    model_type = ModelType(model_type_str)

    if model_type == ModelType.SSM:
        from asr_lab.models.ssm import SSMConfig

        model_config = SSMConfig.from_dict(model_config_dict)
        model = SSMASRModel(model_config)
    elif model_type == ModelType.WHISPER:
        from asr_lab.models.whisper import WhisperConfig

        model_config = WhisperConfig.from_dict(model_config_dict)
        model = WhisperASRModel(model_config)
    elif model_type == ModelType.CONFORMER:
        from asr_lab.models.conformer import ConformerConfig

        model_config = ConformerConfig.from_dict(model_config_dict)
        model = ConformerASRModel(model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def load_fleurs_dataset(
    language: str,
    split: str,
    feature_extractor: MelSpectrogramExtractor,
    tokenizer: CharacterTokenizer,
    cache_dir: str = "/data/razhan/cache/fleurs",
) -> ASRDataset:
    """Load FLEURS dataset for a specific language from raw files."""
    import csv
    import tarfile
    import tempfile
    from huggingface_hub import hf_hub_download

    cache_path = Path(cache_dir) / language / split
    cache_path.mkdir(parents=True, exist_ok=True)

    # Download TSV file
    tsv_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{language}/{split}.tsv",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    # Download and extract audio
    audio_tar_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{language}/audio/{split}.tar.gz",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    # Extract audio files if not already done
    audio_dir = cache_path / "audio"
    if not audio_dir.exists() or len(list(audio_dir.glob("*.wav"))) == 0:
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"    Extracting audio to {audio_dir}...")
        with tarfile.open(audio_tar_path, "r:gz") as tar:
            tar.extractall(audio_dir)

    # Parse TSV and create samples
    # FLEURS TSV format (no header): id, filename, raw_transcription, transcription, chars, duration_samples, gender
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            filename = parts[1]  # e.g., "12741024238657315067.wav"
            transcription = parts[3]  # normalized transcription

            if not filename or not transcription:
                continue

            # Find audio file (extracted to audio_dir/split/filename)
            audio_file = audio_dir / split / filename
            if not audio_file.exists():
                # Try other locations
                for pattern in [audio_dir / filename]:
                    if pattern.exists():
                        audio_file = pattern
                        break

            if not audio_file.exists():
                # Try finding it recursively
                matches = list(audio_dir.rglob(filename))
                if matches:
                    audio_file = matches[0]

            if not audio_file.exists():
                continue

            samples.append(
                ASRSample(
                    audio_path=str(audio_file),
                    text=transcription.lower(),
                    duration=None,  # Will be computed when loading
                    language=language,
                )
            )

    print(f"    Loaded {len(samples)} samples from FLEURS {language}/{split}")
    return ASRDataset(samples, feature_extractor, tokenizer)


def load_librispeech_dataset(
    split: str,
    feature_extractor: MelSpectrogramExtractor,
    tokenizer: CharacterTokenizer,
    manifest_base_dir: str = "/data/razhan/15k_hours",
) -> ASRDataset:
    """Load LibriSpeech dataset from prepared manifests or HuggingFace.

    Args:
        split: Dataset split (e.g., "test.clean", "test.other")
        feature_extractor: Feature extractor for audio
        tokenizer: Tokenizer for text
        manifest_base_dir: Base directory containing prepared manifests
    """
    # Map HuggingFace split names to manifest names
    split_to_manifest = {
        "test.clean": ("librispeech_clean_100", "test"),
        "test.other": ("librispeech_other", "test"),
        "validation.clean": ("librispeech_clean_100", "dev"),
        "validation.other": ("librispeech_other", "dev"),
    }

    # Try to use prepared manifest first
    if split in split_to_manifest:
        dataset_name, manifest_split = split_to_manifest[split]
        manifest_path = Path(manifest_base_dir) / dataset_name / "manifests" / f"{dataset_name}_{manifest_split}.json"

        if manifest_path.exists():
            print(f"    Using prepared manifest: {manifest_path}")
            return ASRDataset.from_manifest(manifest_path, feature_extractor, tokenizer)

    # Fall back to HuggingFace loading with AudioDecoder handling
    try:
        from datasets import load_dataset
        import tempfile
        import numpy as np
        import soundfile as sf

        ds = load_dataset("librispeech_asr", split=split)

        samples = []
        temp_dir = Path(tempfile.mkdtemp(prefix="librispeech_eval_"))
        print(f"    Saving audio to temp dir: {temp_dir}")

        for idx, item in enumerate(ds):
            audio_data = item.get("audio")
            if audio_data is None:
                continue

            # Handle AudioDecoder format (HuggingFace datasets 4.x)
            if hasattr(audio_data, 'metadata'):
                audio_array = np.array(audio_data['array'], dtype=np.float32)
                audio_sr = audio_data.metadata.sample_rate
            # Handle legacy dict format
            elif isinstance(audio_data, dict):
                audio_array = np.array(audio_data.get("array"), dtype=np.float32)
                audio_sr = audio_data.get("sampling_rate", 16000)
            else:
                continue

            # Save to temp file
            audio_path = temp_dir / f"sample_{idx:06d}.wav"
            sf.write(str(audio_path), audio_array, audio_sr, subtype='PCM_16')

            text = item.get("text", "")
            if text:
                samples.append(
                    ASRSample(
                        audio_path=str(audio_path),
                        text=text.lower(),
                        duration=len(audio_array) / audio_sr,
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
    model, full_config = load_model(args.checkpoint)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create feature extractor from config
    features_config = full_config.get("features", {})
    feature_config = FeatureConfig(
        sample_rate=features_config.get("sample_rate", 16000),
        n_mels=features_config.get("n_mels", 80),
    )
    feature_extractor = MelSpectrogramExtractor(feature_config)

    # Create tokenizer from config
    tokenizer_config = full_config.get("tokenizer", {})
    if tokenizer_config.get("type") == "sentencepiece":
        from asr_lab.tokenizers.base import BPETokenizer
        # add_blank=True is required for CTC models (blank token at position 0)
        tokenizer = BPETokenizer(tokenizer_config["model_path"], add_blank=True)
    else:
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
