import pytest

from moelar.backends.mock import MockBackend
from moelar.engine import Engine
from moelar.schema import ChoiceAnswer, MultiAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest

STATE = "Help! My payouts have been failing for 3 days. This is the second time I have written in."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments, payouts, refunds", "technical": "Bugs, outages", "sales": "Pricing"},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"],
    },
    "is_urgent": {"type": "noul", "instructions": "The message conveys urgency"},
    "topics": {"type": "multi", "instructions": "Which topics apply?", "criteria": {"payouts": None, "login": None}},
}


@pytest.fixture
def engine():
    return Engine(MockBackend())


def test_basic_shapes(engine):
    response = engine.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    dept = response.answers["department"]
    assert isinstance(dept, ChoiceAnswer)
    assert set(dept.probabilities) == {"billing", "technical", "sales"}
    assert abs(sum(dept.probabilities.values()) - 1.0) < 1e-3
    assert 0.0 <= dept.confidence <= 1.0
    assert dept.choice == "billing"  # keyword "payouts" appears in the state

    frustration = response.answers["frustration"]
    assert isinstance(frustration, ScoreAnswer)
    assert frustration.legend == {"0": "Calm", "1": "Frustrated but civil", "2": "Very angry"}
    assert 0.0 <= frustration.score <= 2.0

    urgent = response.answers["is_urgent"]
    assert isinstance(urgent, NoulAnswer)
    assert 0.0 <= urgent.noul <= 1.0

    topics = response.answers["topics"]
    assert isinstance(topics, MultiAnswer)
    assert set(topics.probabilities) == {"payouts", "login"}
    assert "payouts" in topics.selected

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens == 5  # 1 choice + 1 score + 1 noul + 2 multi rows
    assert response.model.startswith("moelar-")


def test_determinism(engine):
    request = SystemOneRequest(state=STATE, questions=QUESTIONS)
    first = engine.evaluate(request).model_dump()
    second = engine.evaluate(request).model_dump()
    assert first == second


def test_permutations_report_order_sensitivity(engine):
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, moelar={"permutations": 4})
    dept = engine.evaluate(request).answers["department"]
    assert isinstance(dept, ChoiceAnswer)
    assert dept.order_sensitivity is not None
    assert 0.0 <= dept.order_sensitivity <= 1.0
    assert abs(sum(dept.probabilities.values()) - 1.0) < 1e-3


def test_abstain_margin(engine):
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, moelar={"abstain_margin": 1.0})
    answers = engine.evaluate(request).answers
    assert answers["department"].abstain is True
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, moelar={"abstain_margin": 0.0})
    answers = engine.evaluate(request).answers
    assert answers["department"].abstain is False


def test_complement_constraint(engine):
    questions = {
        "needs_human": {"type": "noul", "instructions": "A human must handle this"},
        "bot_can_resolve": {"type": "noul", "instructions": "A bot can resolve this without a human"},
    }
    request = SystemOneRequest(
        state=STATE,
        questions=questions,
        moelar={"constraints": [{"kind": "complement", "questions": ["needs_human", "bot_can_resolve"]}]},
    )
    answers = engine.evaluate(request).answers
    assert abs(answers["needs_human"].noul + answers["bot_can_resolve"].noul - 1.0) < 1e-3


def test_exclusive_constraint_rescales(engine):
    questions = {f"q{i}": {"type": "noul", "instructions": f"Statement {i} holds"} for i in range(3)}
    request = SystemOneRequest(
        state=STATE,
        questions=questions,
        moelar={"constraints": [{"kind": "exclusive", "questions": list(questions)}]},
    )
    answers = engine.evaluate(request).answers
    assert sum(a.noul for a in answers.values()) <= 1.0 + 1e-6


def test_explain_returns_evidence(engine):
    request = SystemOneRequest(state=STATE, questions=QUESTIONS, moelar={"explain": True})
    answers = engine.evaluate(request).answers
    dept = answers["department"]
    assert isinstance(dept, ChoiceAnswer)
    assert dept.evidence is not None
    assert all(e.effect >= 0 for e in dept.evidence)
    assert all(e.span in STATE for e in dept.evidence)
    topics = answers["topics"]
    assert isinstance(topics, MultiAnswer)
    assert topics.evidence is not None
    assert all(e.span in STATE for e in topics.evidence)


def test_validation_rejects_bad_questions():
    with pytest.raises(ValueError):
        SystemOneRequest(state=STATE, questions={"x": {"type": "choice", "criteria": {"only": None}}})
    with pytest.raises(ValueError):
        SystemOneRequest(state=STATE, questions={"x": {"type": "score", "criteria": ["one"]}})
    with pytest.raises(ValueError):
        SystemOneRequest(state=STATE, questions={"x": {"type": "verdict", "instructions": "?"}})
    with pytest.raises(ValueError):
        SystemOneRequest(
            state=STATE,
            questions={"a": {"type": "noul"}},
            moelar={"constraints": [{"kind": "complement", "questions": ["a", "missing"]}]},
        )


def test_structured_state_and_criteria(engine):
    request = SystemOneRequest(
        state={"ticket": {"subject": "Duplicate charge", "messages": ["I was charged twice"]}},
        questions={
            "refund": {
                "type": "noul",
                "instructions": {"question": "Is a refund requested?", "field": "`ticket.messages[0]`"},
                "criteria": {"true": "Asks for money back", "false": "No refund language"},
            }
        },
    )
    answer = engine.evaluate(request).answers["refund"]
    assert isinstance(answer, NoulAnswer)
