"""Sectioned local test groups."""
from .data_processing import DataProcessingTests
from .dataloader import DataLoaderTests
from .models import ModelTests
from .generation import GenerationTests
from .evaluation import EvaluationTests
from .data_pipeline import DataPipelineTests
from .training import TrainingTests
from .generation_pipeline import GenerationPipelineTests
from .evaluation_pipeline import EvaluationPipelineTests
from .data_utils import DataUtilsTests
from .integration import IntegrationTests
from .load_and_generate import LoadAndGenerateTests
from .virtualiser import VirtualiserTests

__all__ = [
    "DataProcessingTests",
    "DataLoaderTests",
    "ModelTests",
    "GenerationTests",
    "EvaluationTests",
    "DataPipelineTests",
    "TrainingTests",
    "GenerationPipelineTests",
    "EvaluationPipelineTests",
    "DataUtilsTests",
    "IntegrationTests",
    "LoadAndGenerateTests",
    "VirtualiserTests",
]
