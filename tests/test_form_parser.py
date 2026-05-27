from quiz_extractor.parser.form_parser import GoogleFormParser, render_quiz_text


def test_parse_snapshot_groups_sections_and_questions() -> None:
    snapshot = {
        "title": "Web Development Quiz",
        "fetch_status": "rendered_dom",
        "page_url": "https://docs.google.com/forms/d/e/example/viewform",
        "questions": [
            {"title": "HTML Basics", "kind": "section_title"},
            {
                "title": "Which tag creates a paragraph?",
                "kind": "multiple_choice",
                "options": ["<p>", "<div>"],
                "required": True,
            },
            {
                "title": "Explain responsive design.",
                "kind": "long_answer",
                "options": [],
            },
        ],
    }

    quiz = GoogleFormParser.parse_snapshot(
        snapshot,
        form_url="https://docs.google.com/forms/d/e/example/viewform",
        quiz_id="quiz-1",
        fallback_title="Fallback",
    )

    assert quiz["quiz_id"] == "quiz-1"
    assert quiz["title"] == "Web Development Quiz"
    assert quiz["section_titles"] == ["HTML Basics"]
    assert len(quiz["questions"]) == 2
    assert quiz["questions"][0]["question"] == "Which tag creates a paragraph?"
    assert quiz["questions"][0]["options"] == ["<p>", "<div>"]
    assert quiz["questions"][0]["type"] == "multiple_choice"
    assert quiz["questions"][0]["required"] is True
    assert quiz["questions"][0]["section_title"] == "HTML Basics"
    assert "Question 1: Which tag creates a paragraph?" in quiz["quiz_text"]
    assert "Section: HTML Basics" in quiz["quiz_text"]


def test_render_quiz_text_includes_questions() -> None:
    text = render_quiz_text(
        "Sample Quiz",
        [
            {"question": "Q1", "type": "short_answer", "options": []},
            {"question": "Q2", "type": "checkbox", "options": ["A", "B"]},
        ],
        ["Intro"],
    )

    assert "Quiz Title: Sample Quiz" in text
    assert "Section: Intro" in text
    assert "Question 1: Q1" in text
    assert "Options: A, B" in text
