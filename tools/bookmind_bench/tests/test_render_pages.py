from render_pdf import parse_pages


def test_parse_pages_basic_ranges():
    assert parse_pages("1-3,5", 10) == [1, 2, 3, 5]


def test_parse_pages_clamps_to_max():
    assert parse_pages("1-5", 3) == [1, 2, 3]
