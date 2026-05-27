from classroom.forms import build_form_text, extract_form_id, extract_form_questions


def test_extract_form_id_from_view_url() -> None:
    url = "https://docs.google.com/forms/d/e/1FAIpQLSdwPyR8q_-6guGXKlrXyKd0QBNupPuXl4jcM447rLXPT1sbGg/viewform"
    assert extract_form_id(url) == "1FAIpQLSdwPyR8q_-6guGXKlrXyKd0QBNupPuXl4jcM447rLXPT1sbGg"


def test_extract_form_questions_and_render_text() -> None:
    payload = {
        "info": {"title": "Web development quiz"},
        "items": [
            {
                "title": "Which tag creates a paragraph?",
                "questionItem": {
                    "question": {
                        "required": True,
                        "choiceQuestion": {
                            "type": "RADIO",
                            "options": [{"value": "<p>"}, {"value": "<div>"}],
                        },
                    }
                },
            },
            {
                "title": "Explain responsive design.",
                "questionItem": {"question": {"required": False, "textQuestion": {}}},
            },
        ],
    }

    questions = extract_form_questions(payload)
    assert len(questions) == 2
    assert questions[0]["title"] == "Which tag creates a paragraph?"
    assert questions[0]["kind"] == "choice"
    assert questions[0]["options"] == ["<p>", "<div>"]
    assert questions[1]["kind"] == "text"

    rendered = build_form_text("Web development quiz", "https://example.com/form", questions)
    assert "Form Title: Web development quiz" in rendered
    assert "Question 1: Which tag creates a paragraph?" in rendered
    assert "Options: <p>, <div>" in rendered
    assert "Question 2: Explain responsive design." in rendered
