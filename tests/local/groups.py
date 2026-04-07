"""Grouped tests for local suite."""

from tests.local.group_sections import DataProcessingTests
from tests.local.group_sections import DataLoaderTests
from tests.local.group_sections import ModelTests
from tests.local.group_sections import GenerationTests
from tests.local.group_sections import EvaluationTests
from tests.local.group_sections import DataPipelineTests
from tests.local.group_sections import TrainingTests
from tests.local.group_sections import GenerationPipelineTests
from tests.local.group_sections import EvaluationPipelineTests
from tests.local.group_sections import DataUtilsTests
from tests.local.group_sections import IntegrationTests
from tests.local.group_sections import LoadAndGenerateTests
from tests.local.group_sections import VirtualiserTests

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
