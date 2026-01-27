"""
Data preparation pipeline orchestrator.

Coordinates the full data preparation workflow:
1. Download datasets from various sources
2. Load and process text samples
3. Build unified corpus with configurable mixing
4. Export in formats suitable for pretraining

This is the main entry point for preparing AML text data.
"""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from .base import (
    BaseTextDataset,
    ComplianceMode,
    DatasetConfig,
    DatasetRegistry,
    DatasetSource,
    TextSample,
    CorpusStats,
    RESTRICTED_SOURCES,
    get_allowed_sources,
)
from .corpus_builder import (
    UnifiedCorpusBuilder,
    MixingConfig,
    default_aml_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data preparation pipeline."""

    # Base directories
    data_dir: Path = field(default_factory=lambda: Path("./data/aml_corpus"))
    output_dir: Path = field(default_factory=lambda: Path("./data/processed"))
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))

    # Compliance mode - controls which sources are allowed
    compliance_mode: ComplianceMode = ComplianceMode.RESEARCH

    # Dataset selection (will be filtered by compliance_mode)
    enabled_sources: Set[DatasetSource] = field(default_factory=lambda: {
        DatasetSource.FINCEN_FILES,
        DatasetSource.SEC_EDGAR,
        DatasetSource.FRAUDNLP,
        DatasetSource.FATF_TYPOLOGIES,
        DatasetSource.OPENSANCTIONS,
        DatasetSource.VENMO_NOTES,
        DatasetSource.GDELT,
    })

    # Processing options
    auto_download: bool = True
    max_samples_per_source: Optional[int] = None
    min_text_length: int = 20
    max_text_length: int = 50000

    # Corpus building
    mixing_config: Optional[MixingConfig] = None
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Export options
    export_formats: List[str] = field(default_factory=lambda: ["jsonl"])
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05

    # Performance
    num_workers: int = 4
    random_seed: int = 42

    def __post_init__(self):
        """Validate configuration and apply compliance filtering."""
        # Convert paths
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)

        # Validate splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Splits must sum to 1.0, got {total_split}")

        # Apply compliance mode filtering
        allowed = get_allowed_sources(self.compliance_mode)
        original_sources = self.enabled_sources.copy()
        self.enabled_sources = self.enabled_sources & allowed

        # Log which sources were disabled
        disabled = original_sources - self.enabled_sources
        if disabled:
            disabled_names = [s.value for s in disabled]
            logger.warning(
                f"Compliance mode '{self.compliance_mode.value}' disabled sources: "
                f"{', '.join(disabled_names)}"
            )

        # Track disabled sources for audit trail
        self._disabled_sources = disabled
        self._compliance_audit = {
            "mode": self.compliance_mode.value,
            "allowed_sources": [s.value for s in self.enabled_sources],
            "disabled_sources": [s.value for s in disabled],
        }

        # Default mixing config
        if self.mixing_config is None:
            self.mixing_config = MixingConfig(
                source_weights=default_aml_weights(),
            )


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    success: bool
    corpus_stats: Optional[CorpusStats] = None
    output_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    samples_processed: int = 0
    sources_loaded: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "corpus_stats": self.corpus_stats.to_dict() if self.corpus_stats else None,
            "output_files": [str(p) for p in self.output_files],
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.duration_seconds,
            "samples_processed": self.samples_processed,
            "sources_loaded": self.sources_loaded,
        }


class DataPrepPipeline:
    """Main pipeline for preparing AML text data for pretraining.

    This orchestrates the full workflow:
    1. Initialize dataset loaders for each enabled source
    2. Download/load data from each source
    3. Combine into unified corpus with weighted mixing
    4. Export in requested formats

    Example usage:
        ```python
        from rl_money_laundering.dataprep import DataPrepPipeline, PipelineConfig

        config = PipelineConfig(
            data_dir=Path("./data/raw"),
            output_dir=Path("./data/processed"),
            auto_download=True,
        )

        pipeline = DataPrepPipeline(config)
        result = pipeline.run()

        if result.success:
            print(f"Processed {result.samples_processed} samples")
            print(f"Output files: {result.output_files}")
        ```
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._loaders: Dict[DatasetSource, BaseTextDataset] = {}
        self._corpus_builder: Optional[UnifiedCorpusBuilder] = None
        self._callbacks: List[Callable[[str, Any], None]] = []

    def add_callback(self, callback: Callable[[str, Any], None]) -> None:
        """Add a callback for pipeline events.

        Args:
            callback: Function called with (event_name, event_data).
        """
        self._callbacks.append(callback)

    def _emit(self, event: str, data: Any = None) -> None:
        """Emit an event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error for {event}: {e}")

    def _create_loader(self, source: DatasetSource) -> Optional[BaseTextDataset]:
        """Create a loader for the given source.

        Args:
            source: Dataset source to create loader for.

        Returns:
            Loader instance or None if not available.
        """
        # Get loader class from registry
        loader_class = DatasetRegistry.get(source)
        if loader_class is None:
            logger.warning(f"No loader registered for {source.value}")
            return None

        # Create dataset config
        source_dir = self.config.data_dir / source.value
        dataset_config = DatasetConfig(
            data_dir=source_dir,
            auto_download=self.config.auto_download,
            max_samples=self.config.max_samples_per_source,
            min_text_length=self.config.min_text_length,
            random_seed=self.config.random_seed,
        )

        try:
            return loader_class(dataset_config)
        except Exception as e:
            logger.error(f"Failed to create loader for {source.value}: {e}")
            return None

    def _load_source(
        self,
        source: DatasetSource,
    ) -> tuple[DatasetSource, List[TextSample], Optional[str]]:
        """Load samples from a single source.

        Args:
            source: Source to load.

        Returns:
            Tuple of (source, samples, error_message).
        """
        self._emit("source_start", {"source": source.value})

        loader = self._create_loader(source)
        if loader is None:
            return source, [], f"Could not create loader for {source.value}"

        try:
            samples = loader.load()
            self._emit("source_complete", {
                "source": source.value,
                "sample_count": len(samples),
            })
            return source, samples, None
        except Exception as e:
            error_msg = f"Error loading {source.value}: {e}"
            logger.error(error_msg)
            return source, [], error_msg

    def run(self) -> PipelineResult:
        """Run the full data preparation pipeline.

        Returns:
            PipelineResult with statistics and output file paths.
        """
        start_time = datetime.now()
        result = PipelineResult(success=False)

        self._emit("pipeline_start", {"config": str(self.config)})

        try:
            # Create directories
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

            # Load all enabled sources
            all_samples: List[TextSample] = []
            sources_loaded: List[str] = []

            if self.config.num_workers > 1:
                # Parallel loading
                with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                    futures = {
                        executor.submit(self._load_source, source): source
                        for source in self.config.enabled_sources
                    }

                    for future in as_completed(futures):
                        source, samples, error = future.result()
                        if error:
                            result.warnings.append(error)
                        elif samples:
                            all_samples.extend(samples)
                            sources_loaded.append(source.value)
                            logger.info(f"Loaded {len(samples)} samples from {source.value}")
            else:
                # Sequential loading
                for source in self.config.enabled_sources:
                    source, samples, error = self._load_source(source)
                    if error:
                        result.warnings.append(error)
                    elif samples:
                        all_samples.extend(samples)
                        sources_loaded.append(source.value)
                        logger.info(f"Loaded {len(samples)} samples from {source.value}")

            if not all_samples:
                result.errors.append("No samples loaded from any source")
                return result

            logger.info(f"Total samples loaded: {len(all_samples)}")
            self._emit("loading_complete", {"total_samples": len(all_samples)})

            # Build unified corpus
            self._corpus_builder = UnifiedCorpusBuilder(
                mixing_config=self.config.mixing_config,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                random_seed=self.config.random_seed,
            )

            # Add all samples
            for sample in all_samples:
                self._corpus_builder.add_sample(sample)

            # Build corpus
            corpus_samples = self._corpus_builder.build()
            corpus_stats = self._corpus_builder.get_stats()

            logger.info(f"Built corpus with {len(corpus_samples)} samples")
            self._emit("corpus_built", {"sample_count": len(corpus_samples)})

            # Export in requested formats
            output_files: List[Path] = []

            for fmt in self.config.export_formats:
                if fmt == "jsonl":
                    output_path = self.config.output_dir / "aml_corpus.jsonl"
                    self._corpus_builder.export_to_jsonl(output_path)
                    output_files.append(output_path)
                    logger.info(f"Exported to {output_path}")

                elif fmt == "hf":
                    try:
                        dataset = self._corpus_builder.export_to_hf_dataset()
                        hf_path = self.config.output_dir / "aml_corpus_hf"
                        dataset.save_to_disk(str(hf_path))
                        output_files.append(hf_path)
                        logger.info(f"Exported HuggingFace dataset to {hf_path}")
                    except ImportError:
                        result.warnings.append("HuggingFace datasets not available")

                elif fmt == "mlm":
                    train_path = self.config.output_dir / "aml_train.txt"
                    val_path = self.config.output_dir / "aml_val.txt"
                    self._corpus_builder.export_for_mlm(
                        train_path=train_path,
                        val_path=val_path,
                        val_ratio=self.config.val_split,
                    )
                    output_files.extend([train_path, val_path])
                    logger.info(f"Exported MLM files to {train_path}, {val_path}")

                elif fmt == "splits":
                    # Export train/val/test splits
                    for split_name, ratio in [
                        ("train", self.config.train_split),
                        ("val", self.config.val_split),
                        ("test", self.config.test_split),
                    ]:
                        split_samples = self._corpus_builder.get_split(
                            split_name,
                            train_ratio=self.config.train_split,
                            val_ratio=self.config.val_split,
                            test_ratio=self.config.test_split,
                        )
                        split_path = self.config.output_dir / f"aml_{split_name}.jsonl"
                        with open(split_path, "w", encoding="utf-8") as f:
                            for sample in split_samples:
                                f.write(json.dumps(sample.to_dict()) + "\n")
                        output_files.append(split_path)
                        logger.info(f"Exported {split_name} split ({len(split_samples)} samples) to {split_path}")

            # Save corpus statistics
            stats_path = self.config.output_dir / "corpus_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(corpus_stats.to_dict(), f, indent=2)
            output_files.append(stats_path)

            # Update result
            result.success = True
            result.corpus_stats = corpus_stats
            result.output_files = output_files
            result.samples_processed = len(corpus_samples)
            result.sources_loaded = sources_loaded

        except Exception as e:
            result.errors.append(f"Pipeline failed: {e}")
            logger.exception("Pipeline failed")

        finally:
            # Calculate duration
            result.duration_seconds = (datetime.now() - start_time).total_seconds()

            # Save result
            result_path = self.config.output_dir / "pipeline_result.json"
            try:
                self.config.output_dir.mkdir(parents=True, exist_ok=True)
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save pipeline result: {e}")

            self._emit("pipeline_complete", result.to_dict())

        return result

    def run_incremental(
        self,
        new_sources: Optional[Set[DatasetSource]] = None,
    ) -> PipelineResult:
        """Run incremental update with new sources.

        Args:
            new_sources: Additional sources to add. If None, reprocesses all.

        Returns:
            PipelineResult for the incremental run.
        """
        # Load existing corpus if available
        existing_path = self.config.output_dir / "aml_corpus.jsonl"

        if existing_path.exists() and new_sources:
            logger.info("Loading existing corpus for incremental update")
            # TODO: Implement incremental loading
            pass

        # For now, just run full pipeline with updated sources
        if new_sources:
            self.config.enabled_sources = self.config.enabled_sources | new_sources

        return self.run()

    def validate_sources(self) -> Dict[DatasetSource, bool]:
        """Check which sources have data available.

        Returns:
            Dictionary mapping sources to availability status.
        """
        availability = {}

        for source in self.config.enabled_sources:
            loader = self._create_loader(source)
            if loader is None:
                availability[source] = False
            else:
                try:
                    availability[source] = loader.download()
                except Exception:
                    availability[source] = False

        return availability

    def get_source_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each source.

        Returns:
            Dictionary with stats per source.
        """
        if self._corpus_builder is None:
            return {}

        stats = self._corpus_builder.get_stats()
        return stats.source_counts

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        if self.config.cache_dir.exists():
            shutil.rmtree(self.config.cache_dir)
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache cleared")


def run_default_pipeline(
    output_dir: Optional[Path] = None,
    auto_download: bool = True,
    max_samples: Optional[int] = None,
) -> PipelineResult:
    """Run the pipeline with default settings.

    Convenience function for quick corpus building.

    Args:
        output_dir: Output directory for processed data.
        auto_download: Whether to auto-download datasets.
        max_samples: Maximum samples per source (None for no limit).

    Returns:
        PipelineResult with statistics.

    Example:
        ```python
        from rl_money_laundering.dataprep import run_default_pipeline

        result = run_default_pipeline(
            output_dir=Path("./my_corpus"),
            max_samples=10000,
        )
        print(f"Created corpus with {result.samples_processed} samples")
        ```
    """
    config = PipelineConfig(
        output_dir=output_dir or Path("./data/processed"),
        auto_download=auto_download,
        max_samples_per_source=max_samples,
    )

    pipeline = DataPrepPipeline(config)
    return pipeline.run()


def quick_test_pipeline() -> PipelineResult:
    """Run a quick test with synthetic data.

    Useful for testing the pipeline without downloading large datasets.

    Returns:
        PipelineResult from test run.
    """
    config = PipelineConfig(
        output_dir=Path("./data/test_output"),
        auto_download=True,  # Will generate synthetic data if downloads fail
        max_samples_per_source=100,
        enabled_sources={
            DatasetSource.FRAUDNLP,  # Has synthetic fallback
            DatasetSource.VENMO_NOTES,  # Has synthetic fallback
        },
    )

    pipeline = DataPrepPipeline(config)
    return pipeline.run()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AML Text Corpus Data Preparation Pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/aml_corpus"),
        help="Directory for raw data files",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't auto-download missing datasets",
    )
    parser.add_argument(
        "--compliance-mode",
        type=str,
        default="research",
        choices=["research", "strict", "public_only"],
        help="Compliance mode: research (all sources), strict (no leaked data), public_only (only clearly licensed)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per source",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["jsonl"],
        choices=["jsonl", "hf", "mlm", "splits"],
        help="Export formats",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with synthetic data",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.test:
        print("Running quick test pipeline...")
        result = quick_test_pipeline()
    else:
        config = PipelineConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            auto_download=not args.no_download,
            compliance_mode=ComplianceMode(args.compliance_mode),
            max_samples_per_source=args.max_samples,
            export_formats=args.formats,
            num_workers=args.workers,
        )

        pipeline = DataPrepPipeline(config)

        # Add progress callback
        def progress_callback(event: str, data: Any) -> None:
            if event == "source_complete":
                print(f"  Loaded {data['sample_count']} samples from {data['source']}")
            elif event == "corpus_built":
                print(f"  Built corpus with {data['sample_count']} samples")

        pipeline.add_callback(progress_callback)

        print("Running AML data preparation pipeline...")
        result = pipeline.run()

    # Print results
    print("\n" + "=" * 60)
    if result.success:
        print("Pipeline completed successfully!")
        print(f"  Samples processed: {result.samples_processed}")
        print(f"  Sources loaded: {', '.join(result.sources_loaded)}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Output files:")
        for f in result.output_files:
            print(f"    - {f}")
    else:
        print("Pipeline failed!")
        for error in result.errors:
            print(f"  ERROR: {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  WARN: {warning}")
