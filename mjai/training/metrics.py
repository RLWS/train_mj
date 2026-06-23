from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import torch


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算前一准确率。
    
    参数:
        logits (torch.Tensor): 候选动作分数张量。
        labels (torch.Tensor): 标签索引张量。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


RESPONSE_KIND_ORDER = ("pass", "chi", "pon", "daiminkan", "ron", "chankan", "unknown")
RESPONSE_KIND_RANK = {kind: index for index, kind in enumerate(RESPONSE_KIND_ORDER)}


@dataclass(slots=True, frozen=True)
class ResponseTopKSummary:
    samples: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    macro_top1_accuracy: float
    macro_top3_accuracy: float
    macro_top5_accuracy: float
    topk_hits: dict[int, int] = field(default_factory=dict)
    kind_totals: dict[str, int] = field(default_factory=dict)
    kind_topk_hits: dict[str, dict[int, int]] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ResponseBinarySummary:
    actual_pass: int
    actual_non_pass: int
    predicted_pass: int
    predicted_non_pass: int
    pass_precision: float
    pass_recall: float
    pass_f1: float
    non_pass_precision: float
    non_pass_recall: float
    non_pass_f1: float
    false_call_count: int
    missed_call_count: int
    false_call_rate: float
    missed_call_rate: float


@dataclass(slots=True, frozen=True)
class ResponseDiagnostics:
    single_candidate: ResponseTopKSummary
    multi_candidate: ResponseTopKSummary
    binary: ResponseBinarySummary
    candidate_sets: dict[str, ResponseTopKSummary]
    confusion: dict[str, int]


class ResponseMetricsAccumulator:
    """Collect response-specific metrics that are hidden by pass-only accuracy."""

    def __init__(self) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self._single_candidate = _TopKAccumulator()
        self._multi_candidate = _TopKAccumulator()
        self._candidate_sets: dict[str, _TopKAccumulator] = {}
        self._confusion: dict[str, int] = defaultdict(int)
        self._actual_pass = 0
        self._actual_non_pass = 0
        self._predicted_pass = 0
        self._predicted_non_pass = 0
        self._pass_tp = 0
        self._non_pass_tp = 0
        self._false_call_count = 0
        self._missed_call_count = 0

    def update(
        self,
        *,
        label_kind: str,
        predicted_kind: str,
        candidate_kinds: Iterable[str],
        candidate_count: int,
        label_rank: int,
    ) -> None:
        """更新统计或状态累积结果。
        
        参数:
            self (Self): 当前对象实例。
            label_kind (str): 标签动作类型。
            predicted_kind (str): 函数执行所需的输入参数。
            candidate_kinds (Iterable[str]): 相关数据列表。
            candidate_count (int): 相关数量。
            label_rank (int): 函数执行所需的输入参数。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        if candidate_count <= 1:
            self._single_candidate.update(label_kind=label_kind, label_rank=label_rank)
        else:
            self._multi_candidate.update(label_kind=label_kind, label_rank=label_rank)

        candidate_set_name = response_candidate_set_name(candidate_kinds)
        self._candidate_sets.setdefault(candidate_set_name, _TopKAccumulator()).update(
            label_kind=label_kind,
            label_rank=label_rank,
        )

        label_is_pass = label_kind == "pass"
        predicted_is_pass = predicted_kind == "pass"
        if label_is_pass:
            self._actual_pass += 1
        else:
            self._actual_non_pass += 1
        if predicted_is_pass:
            self._predicted_pass += 1
        else:
            self._predicted_non_pass += 1

        if label_is_pass and predicted_is_pass:
            self._pass_tp += 1
        if (not label_is_pass) and (not predicted_is_pass):
            self._non_pass_tp += 1
        if label_is_pass and not predicted_is_pass:
            self._false_call_count += 1
        if (not label_is_pass) and predicted_is_pass:
            self._missed_call_count += 1

        self._confusion[f"{label_kind}->{predicted_kind}"] += 1

    def summary(self) -> ResponseDiagnostics:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            ResponseDiagnostics: 函数执行结果。
        
        异常:
            无。
        """
        pass_precision = _safe_divide(self._pass_tp, self._predicted_pass)
        pass_recall = _safe_divide(self._pass_tp, self._actual_pass)
        non_pass_precision = _safe_divide(self._non_pass_tp, self._predicted_non_pass)
        non_pass_recall = _safe_divide(self._non_pass_tp, self._actual_non_pass)

        return ResponseDiagnostics(
            single_candidate=self._single_candidate.summary(),
            multi_candidate=self._multi_candidate.summary(),
            binary=ResponseBinarySummary(
                actual_pass=self._actual_pass,
                actual_non_pass=self._actual_non_pass,
                predicted_pass=self._predicted_pass,
                predicted_non_pass=self._predicted_non_pass,
                pass_precision=pass_precision,
                pass_recall=pass_recall,
                pass_f1=_f1(pass_precision, pass_recall),
                non_pass_precision=non_pass_precision,
                non_pass_recall=non_pass_recall,
                non_pass_f1=_f1(non_pass_precision, non_pass_recall),
                false_call_count=self._false_call_count,
                missed_call_count=self._missed_call_count,
                false_call_rate=_safe_divide(self._false_call_count, self._actual_pass),
                missed_call_rate=_safe_divide(self._missed_call_count, self._actual_non_pass),
            ),
            candidate_sets={
                name: accumulator.summary()
                for name, accumulator in sorted(self._candidate_sets.items())
            },
            confusion=dict(sorted(self._confusion.items())),
        )


class _TopKAccumulator:
    def __init__(self) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.samples = 0
        self.topk_hits = {1: 0, 3: 0, 5: 0}
        self.kind_totals: dict[str, int] = {}
        self.kind_topk_hits: dict[str, dict[int, int]] = {}

    def update(self, *, label_kind: str, label_rank: int) -> None:
        """更新统计或状态累积结果。
        
        参数:
            self (Self): 当前对象实例。
            label_kind (str): 标签动作类型。
            label_rank (int): 函数执行所需的输入参数。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.samples += 1
        self.kind_totals[label_kind] = self.kind_totals.get(label_kind, 0) + 1
        self.kind_topk_hits.setdefault(label_kind, {1: 0, 3: 0, 5: 0})
        for top_k in (1, 3, 5):
            hit = int(label_rank <= top_k)
            self.topk_hits[top_k] += hit
            self.kind_topk_hits[label_kind][top_k] += hit

    def summary(self) -> ResponseTopKSummary:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            ResponseTopKSummary: 函数执行结果。
        
        异常:
            无。
        """
        return ResponseTopKSummary(
            samples=self.samples,
            top1_accuracy=_safe_divide(self.topk_hits[1], self.samples),
            top3_accuracy=_safe_divide(self.topk_hits[3], self.samples),
            top5_accuracy=_safe_divide(self.topk_hits[5], self.samples),
            macro_top1_accuracy=self._macro_topk_accuracy(1),
            macro_top3_accuracy=self._macro_topk_accuracy(3),
            macro_top5_accuracy=self._macro_topk_accuracy(5),
            topk_hits={top_k: int(hits) for top_k, hits in self.topk_hits.items()},
            kind_totals={kind: int(samples) for kind, samples in self.kind_totals.items()},
            kind_topk_hits={
                kind: {top_k: int(hits) for top_k, hits in hits_by_k.items()}
                for kind, hits_by_k in self.kind_topk_hits.items()
            },
        )

    def _macro_topk_accuracy(self, top_k: int) -> float:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            top_k (int): 前若干候选数量。
        
        返回:
            float: 浮点数结果。
        
        异常:
            无。
        """
        if not self.kind_totals:
            return 0.0
        return sum(
            _safe_divide(self.kind_topk_hits[kind][top_k], samples)
            for kind, samples in self.kind_totals.items()
        ) / len(self.kind_totals)


def response_candidate_set_name(candidate_kinds: Iterable[str]) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        candidate_kinds (Iterable[str]): 相关数据列表。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    unique_kinds = sorted(
        {str(kind) for kind in candidate_kinds},
        key=lambda kind: RESPONSE_KIND_RANK.get(kind, len(RESPONSE_KIND_ORDER)),
    )
    if unique_kinds == ["pass"]:
        return "pass_only"
    return "+".join(unique_kinds) if unique_kinds else "empty"


def response_selection_score(diagnostics: ResponseDiagnostics | None) -> float | None:
    """执行与函数名对应的业务逻辑。
    
    参数:
        diagnostics (ResponseDiagnostics | None): 相关数据列表。
    
    返回:
        float | None: 函数执行结果。
    
    异常:
        无。
    """
    if diagnostics is None:
        return None
    binary = diagnostics.binary
    # Prefer a useful respond/pass decision, while discouraging false calls.
    return binary.non_pass_f1 - binary.false_call_rate


def combine_response_diagnostics(
    diagnostics: Iterable[ResponseDiagnostics | None],
) -> ResponseDiagnostics | None:
    """Combine per-rank response diagnostics into a single exact summary."""

    items = [item for item in diagnostics if item is not None]
    if not items:
        return None

    candidate_set_names = sorted({name for item in items for name in item.candidate_sets})
    confusion: dict[str, int] = defaultdict(int)
    for item in items:
        for key, value in item.confusion.items():
            confusion[key] += int(value)

    return ResponseDiagnostics(
        single_candidate=_combine_topk_summaries(item.single_candidate for item in items),
        multi_candidate=_combine_topk_summaries(item.multi_candidate for item in items),
        binary=_combine_binary_summaries(item.binary for item in items),
        candidate_sets={
            name: _combine_topk_summaries(
                item.candidate_sets[name]
                for item in items
                if name in item.candidate_sets
            )
            for name in candidate_set_names
        },
        confusion=dict(sorted(confusion.items())),
    )


def _combine_topk_summaries(summaries: Iterable[ResponseTopKSummary]) -> ResponseTopKSummary:
    items = list(summaries)
    total_samples = sum(int(item.samples) for item in items)
    topk_hits = {top_k: 0 for top_k in (1, 3, 5)}
    kind_totals: dict[str, int] = defaultdict(int)
    kind_topk_hits: dict[str, dict[int, int]] = defaultdict(lambda: {1: 0, 3: 0, 5: 0})

    for item in items:
        for top_k in (1, 3, 5):
            topk_hits[top_k] += _topk_hit_count(item, top_k)

        if item.kind_totals:
            for kind, samples in item.kind_totals.items():
                kind_totals[kind] += int(samples)
            for kind, hits_by_k in item.kind_topk_hits.items():
                for top_k in (1, 3, 5):
                    kind_topk_hits[kind][top_k] += int(hits_by_k.get(top_k, 0))
        elif item.samples > 0:
            kind_totals["__all__"] += int(item.samples)
            for top_k in (1, 3, 5):
                kind_topk_hits["__all__"][top_k] += _topk_hit_count(item, top_k)

    return ResponseTopKSummary(
        samples=total_samples,
        top1_accuracy=_safe_divide(topk_hits[1], total_samples),
        top3_accuracy=_safe_divide(topk_hits[3], total_samples),
        top5_accuracy=_safe_divide(topk_hits[5], total_samples),
        macro_top1_accuracy=_macro_topk_accuracy_from_counts(kind_totals, kind_topk_hits, 1),
        macro_top3_accuracy=_macro_topk_accuracy_from_counts(kind_totals, kind_topk_hits, 3),
        macro_top5_accuracy=_macro_topk_accuracy_from_counts(kind_totals, kind_topk_hits, 5),
        topk_hits=dict(topk_hits),
        kind_totals=dict(sorted(kind_totals.items())),
        kind_topk_hits={
            kind: dict(hits_by_k)
            for kind, hits_by_k in sorted(kind_topk_hits.items())
        },
    )


def _topk_hit_count(summary: ResponseTopKSummary, top_k: int) -> int:
    if summary.topk_hits:
        return int(summary.topk_hits.get(top_k, 0))
    accuracy = {
        1: summary.top1_accuracy,
        3: summary.top3_accuracy,
        5: summary.top5_accuracy,
    }[top_k]
    return int(round(float(accuracy) * int(summary.samples)))


def _macro_topk_accuracy_from_counts(
    kind_totals: dict[str, int],
    kind_topk_hits: dict[str, dict[int, int]],
    top_k: int,
) -> float:
    values = [
        _safe_divide(kind_topk_hits.get(kind, {}).get(top_k, 0), samples)
        for kind, samples in kind_totals.items()
        if samples > 0
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _combine_binary_summaries(
    summaries: Iterable[ResponseBinarySummary],
) -> ResponseBinarySummary:
    items = list(summaries)
    actual_pass = sum(int(item.actual_pass) for item in items)
    actual_non_pass = sum(int(item.actual_non_pass) for item in items)
    predicted_pass = sum(int(item.predicted_pass) for item in items)
    predicted_non_pass = sum(int(item.predicted_non_pass) for item in items)
    false_call_count = sum(int(item.false_call_count) for item in items)
    missed_call_count = sum(int(item.missed_call_count) for item in items)
    pass_tp = actual_pass - false_call_count
    non_pass_tp = actual_non_pass - missed_call_count
    pass_precision = _safe_divide(pass_tp, predicted_pass)
    pass_recall = _safe_divide(pass_tp, actual_pass)
    non_pass_precision = _safe_divide(non_pass_tp, predicted_non_pass)
    non_pass_recall = _safe_divide(non_pass_tp, actual_non_pass)
    return ResponseBinarySummary(
        actual_pass=actual_pass,
        actual_non_pass=actual_non_pass,
        predicted_pass=predicted_pass,
        predicted_non_pass=predicted_non_pass,
        pass_precision=pass_precision,
        pass_recall=pass_recall,
        pass_f1=_f1(pass_precision, pass_recall),
        non_pass_precision=non_pass_precision,
        non_pass_recall=non_pass_recall,
        non_pass_f1=_f1(non_pass_precision, non_pass_recall),
        false_call_count=false_call_count,
        missed_call_count=missed_call_count,
        false_call_rate=_safe_divide(false_call_count, actual_pass),
        missed_call_rate=_safe_divide(missed_call_count, actual_non_pass),
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        numerator (int | float): 函数执行所需的输入参数。
        denominator (int | float): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        precision (float): 函数执行所需的输入参数。
        recall (float): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
