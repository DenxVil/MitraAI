"""
🤖 Mitra AI - Data Collector
Download and manage open datasets for training.
Coded by Denvil with love 🤍
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: str  # HuggingFace hub path
    description: str
    size_mb: Optional[float] = None
    num_samples: Optional[int] = None
    task: str = "general"  # general, math, code, reasoning, etc.
    downloaded: bool = False
    local_path: Optional[str] = None


# Available open datasets for training
AVAILABLE_DATASETS: Dict[str, DatasetInfo] = {
    "openhermes": DatasetInfo(
        name="OpenHermes 2.5",
        source="teknium/OpenHermes-2.5",
        description="High-quality instruction-following data",
        task="general",
    ),
    "metamathqa": DatasetInfo(
        name="MetaMathQA",
        source="meta-math/MetaMathQA",
        description="Mathematical reasoning with step-by-step solutions",
        task="math",
    ),
    "codealpaca": DatasetInfo(
        name="Code Alpaca",
        source="sahil2801/CodeAlpaca-20k",
        description="Code instruction data",
        task="code",
    ),
    "wizardlm": DatasetInfo(
        name="WizardLM",
        source="WizardLM/WizardLM_evol_instruct_V2_196k",
        description="Evolved instruction data",
        task="general",
    ),
    "orca": DatasetInfo(
        name="OpenOrca",
        source="Open-Orca/OpenOrca",
        description="Reasoning data with explanations",
        task="reasoning",
    ),
    "slimorca": DatasetInfo(
        name="SlimOrca",
        source="Open-Orca/SlimOrca",
        description="Curated subset of OpenOrca",
        task="reasoning",
    ),
    "platypus": DatasetInfo(
        name="Platypus",
        source="garage-bAInd/Open-Platypus",
        description="STEM and logic reasoning",
        task="reasoning",
    ),
    "gsm8k_train": DatasetInfo(
        name="GSM8K Training",
        source="gsm8k",
        description="Grade school math problems",
        task="math",
    ),
}


class DataCollector:
    """
    Collect and manage datasets for training.

    Features:
    - Download from HuggingFace Hub
    - Local caching
    - Dataset mixing
    - Streaming support
    """

    def __init__(
        self,
        cache_dir: str = "./data/datasets",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, Any] = {}
        self._downloaded: Dict[str, DatasetInfo] = {}

    def list_available(self) -> List[DatasetInfo]:
        """List all available datasets."""
        return list(AVAILABLE_DATASETS.values())

    def list_by_task(self, task: str) -> List[DatasetInfo]:
        """List datasets by task type."""
        return [
            ds for ds in AVAILABLE_DATASETS.values()
            if ds.task == task
        ]

    async def download(
        self,
        dataset_name: str,
        split: str = "train",
        streaming: bool = False,
    ) -> Any:
        """
        Download a dataset.

        Args:
            dataset_name: Name of dataset to download
            split: Dataset split to download
            streaming: Whether to use streaming mode

        Returns:
            The dataset object
        """
        info = AVAILABLE_DATASETS.get(dataset_name)
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(
            "downloading_dataset",
            name=dataset_name,
            source=info.source,
            streaming=streaming,
        )

        try:
            from datasets import load_dataset

            # Download from HuggingFace
            dataset = load_dataset(
                info.source,
                split=split,
                streaming=streaming,
                cache_dir=str(self.cache_dir),
            )

            # Update info
            info.downloaded = True
            info.local_path = str(self.cache_dir / dataset_name)
            if not streaming:
                info.num_samples = len(dataset)

            self._datasets[dataset_name] = dataset
            self._downloaded[dataset_name] = info

            logger.info(
                "dataset_downloaded",
                name=dataset_name,
                samples=info.num_samples,
            )

            return dataset

        except Exception as e:
            logger.error("dataset_download_failed", name=dataset_name, error=str(e))
            raise

    async def download_multiple(
        self,
        dataset_names: List[str],
        split: str = "train",
    ) -> Dict[str, Any]:
        """Download multiple datasets."""
        results = {}
        for name in dataset_names:
            try:
                results[name] = await self.download(name, split)
            except Exception as e:
                logger.error("dataset_download_failed", name=name, error=str(e))
        return results

    async def download_by_task(
        self,
        task: str,
        split: str = "train",
    ) -> Dict[str, Any]:
        """Download all datasets for a task."""
        datasets = self.list_by_task(task)
        names = [ds.name.lower().replace(" ", "_") for ds in datasets]
        return await self.download_multiple(names, split)

    def get_dataset(self, name: str) -> Optional[Any]:
        """Get a downloaded dataset."""
        return self._datasets.get(name)

    def get_info(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset info."""
        return self._downloaded.get(name) or AVAILABLE_DATASETS.get(name)

    async def mix_datasets(
        self,
        datasets: Dict[str, float],
        total_samples: Optional[int] = None,
    ) -> Any:
        """
        Mix multiple datasets with specified ratios.

        Args:
            datasets: Dict mapping dataset names to sampling ratios
            total_samples: Total number of samples in mixed dataset

        Returns:
            Mixed dataset
        """
        from datasets import concatenate_datasets

        mixed_parts = []
        total_ratio = sum(datasets.values())

        for name, ratio in datasets.items():
            ds = self._datasets.get(name)
            if ds is None:
                logger.warning("dataset_not_loaded", name=name)
                continue

            # Calculate number of samples for this dataset
            if total_samples:
                n_samples = int(total_samples * ratio / total_ratio)
            else:
                n_samples = int(len(ds) * ratio / total_ratio)

            # Sample from dataset
            if n_samples < len(ds):
                sampled = ds.shuffle().select(range(n_samples))
            else:
                sampled = ds

            mixed_parts.append(sampled)

        if not mixed_parts:
            raise ValueError("No datasets to mix")

        mixed = concatenate_datasets(mixed_parts)
        mixed = mixed.shuffle()

        logger.info("datasets_mixed", total_samples=len(mixed))

        return mixed

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about downloaded datasets."""
        return {
            "downloaded_count": len(self._downloaded),
            "total_samples": sum(
                info.num_samples or 0
                for info in self._downloaded.values()
            ),
            "datasets": {
                name: {
                    "samples": info.num_samples,
                    "task": info.task,
                }
                for name, info in self._downloaded.items()
            },
        }

    def clear_cache(self) -> None:
        """Clear the dataset cache."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._datasets.clear()
        self._downloaded.clear()
        logger.info("cache_cleared")
