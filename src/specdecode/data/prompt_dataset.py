"""Curated evaluation and training prompt datasets across multiple workload categories."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PromptItem:
    id: str
    category: str
    prompt: str
    target_output_length: int = 64


# Comprehensive benchmark dataset representing diverse deployment workloads
BENCHMARK_PROMPTS: List[PromptItem] = [
    PromptItem(
        id="qa_01",
        category="General QA",
        prompt="Explain the difference between a process and a thread in modern operating systems.",
        target_output_length=64,
    ),
    PromptItem(
        id="code_01",
        category="Code Generation",
        prompt="Write a Python function to compute the longest increasing subsequence in an array with O(N log N) complexity.",
        target_output_length=80,
    ),
    PromptItem(
        id="math_01",
        category="Mathematical Reasoning",
        prompt="Solve the system of equations step by step: 3x + 2y = 12 and 5x - y = 7.",
        target_output_length=64,
    ),
    PromptItem(
        id="structured_01",
        category="Structured Output",
        prompt="Generate a JSON object describing an edge IoT sensor cluster with fields for cluster_id, nodes, telemetry_metrics, and status.",
        target_output_length=70,
    ),
    PromptItem(
        id="technical_01",
        category="Technical Explanations",
        prompt="Explain how KV cache works in transformer inference and why memory bandwidth is the primary bottleneck during decoding.",
        target_output_length=80,
    ),
    PromptItem(
        id="short_01",
        category="Short Interactive",
        prompt="What is the capital of Australia?",
        target_output_length=20,
    ),
    PromptItem(
        id="instruction_01",
        category="Instruction Following",
        prompt="List five concrete strategies to reduce latency in real-time inference systems, ordered by ease of implementation.",
        target_output_length=80,
    ),
    PromptItem(
        id="long_context_01",
        category="Long Context",
        prompt=(
            "Context: In computing, a cache is a hardware or software component that stores data so that future requests "
            "for that data can be served faster. The data stored in a cache might be the result of an earlier computation "
            "or a copy of data stored elsewhere. A cache hit occurs when the requested data can be found in a cache, "
            "while a cache miss occurs when it cannot. Cache hits are served by reading data from the cache, which is faster "
            "than recomputing a result or reading from a slower data store; thus, the more requests that can be served from "
            "the cache, the faster the system performs. To be cost-effective and to enable efficient use of data, caches "
            "must be relatively small. Nevertheless, caches have proven effective in many areas of computing, because "
            "typical computer applications access data with a high degree of locality of reference. Such access patterns "
            "exhibit temporal locality, where data is requested again soon after being requested, and spatial locality, "
            "where data located physically close to previously requested data is requested.\n\n"
            "Question: Based on the provided text, distinguish temporal locality from spatial locality and explain why caches remain small."
        ),
        target_output_length=75,
    ),
]


def get_prompts_by_category() -> Dict[str, List[PromptItem]]:
    """Group benchmark prompts by category."""
    categorized: Dict[str, List[PromptItem]] = {}
    for item in BENCHMARK_PROMPTS:
        categorized.setdefault(item.category, []).append(item)
    return categorized
