from moelars.render import SENTINEL, SYSTEM_PROMPT, TEMPLATES, compose_prompt, fence_state


def test_sentinel_inside_state_stays_inside_the_fence():
    state = f"before {SENTINEL} after"
    for template in TEMPLATES.values():
        prefix, suffix = compose_prompt(template, state, "QUESTION: x")
        assert prefix.endswith(fence_state(state) + "\n\n")
        assert suffix.startswith("QUESTION: x")
        assert SENTINEL not in suffix


def test_prompt_is_unchanged_for_ordinary_state():
    for template in TEMPLATES.values():
        prefix, suffix = compose_prompt(template, "plain state", "QUESTION: x")
        expected = template(SYSTEM_PROMPT, fence_state("plain state") + "\n\nQUESTION: x") + "Answer:"
        assert prefix + suffix == expected
