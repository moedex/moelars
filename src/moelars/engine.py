"""The engine: request in, typed answers out.

1. Plan rows (base, permutation, ablation) from the request.
2. Group rows by prefix and ask the backend for label logits.
3. Reduce each question's rows to one answer: calibrated probabilities, confidence,
   order sensitivity, abstention, evidence.
4. Apply declared constraints across noul answers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from moelars.backends.base import Backend
from moelars.calibration import Calibrator, fused_probability
from moelars.heads import PointerHeadScorer
from moelars.labels import assign_labels
from moelars.primitives import (
    choice_confidence,
    expected_score,
    order_sensitivity,
    round_probs,
    score_confidence,
    sigmoid,
    softmax,
    top_margin,
    total_variation,
)
from moelars.render import Row, compose_prompt, plan_rows, render_content
from moelars.schema import (
    MAX_CHOICE_OPTIONS,
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    Constraint,
    Evidence,
    ModelCard,
    MultiAnswer,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

MODEL_ALIAS = "moelars-latest"
# The official SDKs send "jev-latest" unless told otherwise, so it names the loaded model too;
# that keeps a base-URL swap sufficient. Any other name is refused rather than silently served.
DEFAULT_ALIASES = (MODEL_ALIAS, "jev-latest")
MAX_EVIDENCE = 3
# Per-request budgets, checked before any model pass. A schema-valid request can plan
# thousands of rows (a 255-option multi question with explain=true and 24 state units is
# 6,375), so these bound what one caller can ask of the model.
DEFAULT_MAX_ROWS = 512
DEFAULT_MAX_INPUT_TOKENS = 32768
# Dykstra passes over overlapping constraints; random cases converge in under 100.
CONSTRAINT_ROUNDS = 2000
CONSTRAINT_TOLERANCE = 1e-10


class BudgetError(ValueError):
    """A request that would exceed the engine's row or input-token budget."""


class UnknownModelError(ValueError):
    """A request naming a model this engine does not serve."""


@dataclass
class Scored:
    row: Row
    probs: np.ndarray
    logits: np.ndarray


class Engine:
    def __init__(
        self,
        backend: Backend,
        calibrator: Calibrator | None = None,
        version: str = "0.0.1",
        head: PointerHeadScorer | None = None,
        max_rows: int | None = DEFAULT_MAX_ROWS,
        max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS,
    ) -> None:
        """`max_rows` and `max_input_tokens` bound each `evaluate` call; None disables a budget."""
        self.backend = backend
        self.max_rows = max_rows
        self.max_input_tokens = max_input_tokens
        self.calibrator = calibrator or Calibrator()
        self.version = version
        self.head = head
        if head is not None and not hasattr(backend, "label_logits_with_features"):
            raise ValueError(f"backend {backend.name!r} cannot supply hidden states for a pointer head")
        self.labels = assign_labels(MAX_CHOICE_OPTIONS, backend.is_single_token)

    # ----------------------------------------------------------------- public

    @property
    def model_id(self) -> str:
        return f"moelars-{self.version}+{self.backend.name}:{self.backend.model_name}"

    def models(self) -> list[ModelCard]:
        return [
            ModelCard(name=MODEL_ALIAS, description=f"Alias for {self.model_id}", release_date="2026-09-22"),
            ModelCard(
                name=self.model_id,
                description=f"moe-LARS {self.version} on {self.backend.name} backend ({self.backend.model_name})",
                release_date="2026-09-22",
            ),
        ]

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        if request.model not in (*DEFAULT_ALIASES, self.model_id):
            raise UnknownModelError(f"model {request.model!r} is not served here; use {MODEL_ALIAS!r} "
                                    f"or {self.model_id!r}")
        rows = plan_rows(request, self.labels)
        if self.max_rows is not None and len(rows) > self.max_rows:
            raise BudgetError(f"request plans {len(rows)} model rows, over the budget of {self.max_rows}; "
                              "use fewer questions, options, permutations, or turn off explain")
        usage = self._usage(request, rows)
        if self.max_input_tokens is not None and usage.input_tokens > self.max_input_tokens:
            raise BudgetError(f"request needs {usage.input_tokens} input tokens, over the budget of "
                              f"{self.max_input_tokens}; shorten the state or turn off explain")
        scored = self._score(request, rows)
        answers = self._reduce(request, scored)
        self._apply_constraints(answers, request.moelars.constraints)
        return SystemOneResponse(model=self.model_id, answers=answers, usage=usage)

    def raw_logits(self, state: object, question_id: str, question: object) -> tuple[tuple[str, ...], np.ndarray]:
        """Uncalibrated label logits for one question. Used by calibration and evals."""
        request = SystemOneRequest(state=state, questions={question_id: question})  # type: ignore[arg-type]
        rows = [r for r in plan_rows(request, self.labels) if r.variant == "base"]
        scored = self._score(request, rows)
        row = scored[0]
        return row.row.keys, row.logits

    # ----------------------------------------------------------------- scoring

    def _score(self, request: SystemOneRequest, rows: list[Row]) -> list[Scored]:
        base_state = render_content(request.state)
        template = self.backend.template()
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, row in enumerate(rows):
            state_text = row.state_text if row.state_text is not None else base_state
            prefix, suffix = compose_prompt(template, state_text, row.suffix_body)
            groups[prefix].append((index, suffix))

        scored: list[Scored | None] = [None] * len(rows)
        for prefix, members in groups.items():
            suffixes = [s for _, s in members]
            labels = [rows[i].labels for i, _ in members]
            if self.head is None:
                logits = self.backend.label_logits(prefix, suffixes, labels)
            else:
                logits = []
                ends = [rows[i].option_ends for i, _ in members]
                features = self.backend.label_logits_with_features(prefix, suffixes, labels, ends)  # type: ignore[attr-defined]
                for (index, _), (z, h_ans, h_opt) in zip(members, features, strict=True):
                    kind = "noul" if rows[index].kind == "multi" else rows[index].kind
                    logits.append(self.head.adjust(z, self.head.project(h_ans), self.head.project(h_opt), kind))
            for (index, _), row_logits in zip(members, logits, strict=True):
                row = rows[index]
                temperature = self.calibrator.temperature_for(row.kind)
                scored[index] = Scored(row, softmax(row_logits, temperature), np.asarray(row_logits))
        return [s for s in scored if s is not None]

    # ----------------------------------------------------------------- reduction

    def _reduce(self, request: SystemOneRequest, scored: list[Scored]) -> dict[str, Answer]:
        by_question: dict[str, list[Scored]] = defaultdict(list)
        for item in scored:
            by_question[item.row.question_id].append(item)

        answers: dict[str, Answer] = {}
        options = request.moelars
        for qid, question in request.questions.items():
            items = by_question[qid]
            base = [s for s in items if s.row.variant == "base" or s.row.variant.startswith("base:")]
            perms = [s for s in items if s.row.variant.startswith("perm:")]
            ablations = [s for s in items if s.row.variant.startswith("ablate:")]

            if question.type == "noul":
                p_yes = self.noul_prob(self._yes_logit(base[0]), features=options.features.get(qid))
                answer: Answer = NoulAnswer(noul=round(p_yes, 4))
                if options.abstain_margin is not None:
                    answer.abstain = abs(p_yes - 0.5) * 2 < options.abstain_margin
                effects = [(s.row.ablated_span, abs(self.noul_prob(self._yes_logit(s)) - p_yes)) for s in ablations]
                answer.evidence = self._evidence(effects, options.explain)

            elif question.type == "choice":
                assert isinstance(question, ChoiceQuestion)
                keys = tuple(question.criteria)
                dists = [self._realign(s, keys) for s in base + perms]
                mean = np.mean(np.stack(dists), axis=0)
                probs = round_probs(keys, mean)
                answer = ChoiceAnswer(
                    choice=keys[int(mean.argmax())],
                    probabilities=probs,
                    confidence=round(choice_confidence(mean), 4),
                )
                if perms:
                    answer.order_sensitivity = round(order_sensitivity(dists), 4)
                if options.abstain_margin is not None:
                    answer.abstain = top_margin(mean) < options.abstain_margin
                effects = [(s.row.ablated_span, total_variation(self._realign(s, keys), mean)) for s in ablations]
                answer.evidence = self._evidence(effects, options.explain)

            elif question.type == "score":
                assert isinstance(question, ScoreQuestion)
                probs_vec = base[0].probs
                keys = base[0].row.keys
                legend = {str(i): (c if c is not None else f"level {i}") for i, c in enumerate(question.criteria)}
                answer = ScoreAnswer(
                    score=round(expected_score(probs_vec), 4),
                    legend=legend,
                    probabilities=round_probs(keys, probs_vec),
                    confidence=round(score_confidence(probs_vec), 4),
                )
                if options.abstain_margin is not None:
                    answer.abstain = top_margin(probs_vec) < options.abstain_margin
                effects = [(s.row.ablated_span, total_variation(s.probs, probs_vec)) for s in ablations]
                answer.evidence = self._evidence(effects, options.explain)

            else:  # multi: one yes/no row per option, variants "base:<key>" and "ablate:<i>:<key>"
                per_option: dict[str, float] = {}
                for s in base:
                    key = s.row.variant.split(":", 1)[1]
                    per_option[key] = round(self.noul_prob(self._yes_logit(s), kind="multi"), 4)
                answer = MultiAnswer(
                    probabilities=per_option,
                    selected=[k for k, p in per_option.items() if p >= 0.5],
                )
                by_unit: dict[str, list[float]] = defaultdict(list)
                spans: dict[str, str | None] = {}
                for s in ablations:
                    _, unit, key = s.row.variant.split(":", 2)
                    spans[unit] = s.row.ablated_span
                    ablated = self.noul_prob(self._yes_logit(s), kind="multi")
                    by_unit[unit].append(abs(ablated - per_option.get(key, 0.0)))
                effects = [(spans[unit], float(np.mean(values))) for unit, values in by_unit.items()]
                answer.evidence = self._evidence(effects, options.explain)

            answers[qid] = answer
        return answers

    @staticmethod
    def _yes_logit(item: Scored) -> float:
        """Raw yes-minus-no logit for a two-label row."""
        return float(item.logits[0] - item.logits[1])

    def noul_prob(self, yes_logit: float, kind: str = "noul", features: dict[str, float] | None = None) -> float:
        """P(yes) from the raw yes-minus-no logit.

        With caller evidence and a fusion fitted on the same feature names, P(yes) is the
        fused logistic. Otherwise a fitted Platt pair (a, b) gives sigmoid(a * z + b), or the
        kind's temperature applies: sigmoid(z / T), the special case a = 1/T, b = 0.
        """
        fusion = self.calibrator.fusion_for(kind, features)
        if fusion is not None:
            return fused_probability(fusion, yes_logit, features or {})
        platt = self.calibrator.platt_for(kind)
        if platt is None:
            return sigmoid(yes_logit / self.calibrator.temperature_for(kind))
        a, b = platt
        return sigmoid(a * yes_logit + b)

    @staticmethod
    def _realign(item: Scored, canonical_keys: tuple[str, ...]) -> np.ndarray:
        """Map a row's distribution (in presented order) back to canonical key order."""
        by_key = dict(zip(item.row.keys, item.probs, strict=True))
        return np.asarray([by_key[k] for k in canonical_keys], dtype=np.float64)

    @staticmethod
    def _evidence(effects: list[tuple[str | None, float]], enabled: bool) -> list[Evidence] | None:
        """Top spans by how far the answer moved when each was removed from the state."""
        if not enabled or not effects:
            return None
        ranked = sorted(((span or "", float(value)) for span, value in effects), key=lambda pair: -pair[1])
        return [Evidence(span=span, effect=round(value, 4)) for span, value in ranked[:MAX_EVIDENCE] if value > 0]

    # ----------------------------------------------------------------- constraints

    @staticmethod
    def _apply_constraints(answers: dict[str, Answer], constraints: list[Constraint]) -> None:
        """Enforce the declared constraints on noul answers, jointly.

        Constraints that share no question are applied as declared: a complement averages
        the pair, an exclusive group over 1 is rescaled. When constraints share a question,
        one pass in order can undo an earlier one (exclusive over a, b, c, then complement
        over a, b), so the answers become the nearest probabilities, in least squares, that
        satisfy all of them (Dykstra's alternating projections). Rounding comes last:
        exclusive members round down so their sum cannot exceed 1, and a complement's
        second answer is 1 minus its rounded first.
        """
        applicable = [c for c in constraints if all(isinstance(answers[q], NoulAnswer) for q in c.questions)]
        if not applicable:
            return
        p = {q: float(answers[q].noul) for c in applicable for q in c.questions}  # type: ignore[union-attr]
        members = [q for c in applicable for q in c.questions]
        if len(members) == len(set(members)):
            for constraint in applicable:
                qs = constraint.questions
                if constraint.kind == "complement":
                    x = (p[qs[0]] + 1.0 - p[qs[1]]) / 2.0
                    p[qs[0]], p[qs[1]] = x, 1.0 - x
                else:
                    total = sum(p[q] for q in qs)
                    if total > 1.0:
                        for q in qs:
                            p[q] /= total
        else:
            p = _project_constraints(p, applicable)
        exclusive = {q for c in applicable if c.kind == "exclusive" for q in c.questions}
        rounded = {q: (math.floor(v * 1e4) / 1e4 if q in exclusive else round(v, 4)) for q, v in p.items()}
        for constraint in applicable:
            if constraint.kind == "complement":
                first, second = constraint.questions
                rounded[second] = round(1.0 - rounded[first], 4)
        for q, value in rounded.items():
            answers[q].noul = value  # type: ignore[union-attr]

    # ----------------------------------------------------------------- usage

    def _usage(self, request: SystemOneRequest, rows: list[Row]) -> Usage:
        template = self.backend.template()
        base_state = render_content(request.state)
        seen_prefixes: set[str] = set()
        tokens = 0
        for row in rows:
            state_text = row.state_text if row.state_text is not None else base_state
            prefix, suffix = compose_prompt(template, state_text, row.suffix_body)
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                tokens += self.backend.count_tokens(prefix)
            tokens += self.backend.count_tokens(suffix)
        return Usage(input_tokens=tokens, output_tokens=len(rows))


def _project_constraints(p: dict[str, float], constraints: list[Constraint]) -> dict[str, float]:
    """Nearest point to `p` in least squares with every constraint and 0 <= p <= 1 satisfied.

    Dykstra's algorithm over the sets: each complement's line p_a + p_b = 1, each exclusive
    group's half-space sum <= 1, and the unit box. Every step is an orthogonal projection;
    the per-set increments make the limit the projection onto their intersection.
    """
    keys = list(p)
    index = {q: i for i, q in enumerate(keys)}
    x = np.array([p[q] for q in keys], dtype=np.float64)
    sets = [(c.kind, [index[q] for q in c.questions]) for c in constraints] + [("box", [])]
    increments = [np.zeros_like(x) for _ in sets]
    for _ in range(CONSTRAINT_ROUNDS):
        previous = x.copy()
        for k, (kind, idx) in enumerate(sets):
            y = x + increments[k]
            z = y.copy()
            if kind == "complement":
                shift = (y[idx[0]] + y[idx[1]] - 1.0) / 2.0
                z[idx] -= shift
            elif kind == "exclusive":
                total = y[idx].sum()
                if total > 1.0:
                    z[idx] -= (total - 1.0) / len(idx)
            else:
                z = np.clip(y, 0.0, 1.0)
            increments[k] = y - z
            x = z
        if np.abs(x - previous).max() < CONSTRAINT_TOLERANCE:
            break
    return {q: float(x[i]) for q, i in index.items()}
