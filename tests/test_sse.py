from agent.providers.sse import SSEFrame, SSEParser


def parse(*chunks: bytes) -> list[SSEFrame]:
    p = SSEParser()
    out: list[SSEFrame] = []
    for c in chunks:
        out.extend(p.feed(c))
    out.extend(p.close())
    return out


def test_single_frame():
    frames = parse(b"event: message_stop\ndata: {}\n\n")
    assert frames == [SSEFrame(event="message_stop", data="{}")]


def test_multiple_data_lines_join_with_newline():
    frames = parse(b"event: x\ndata: line1\ndata: line2\n\n")
    assert frames[0].data == "line1\nline2"


def test_comments_are_skipped():
    frames = parse(b": heartbeat\nevent: ping\ndata: {}\n\n")
    assert len(frames) == 1
    assert frames[0].event == "ping"


def test_event_without_data_dispatches_nothing():
    assert parse(b"event: lonely\n\n") == []


def test_byte_at_a_time_matches_whole_stream():
    """The adversarial case: every possible chunk boundary, at once."""
    raw = (
        b"event: content_block_delta\n"
        b'data: {"index":1,"text":"hello"}\n'
        b"\n"
        b": keepalive\n"
        b"\n"
        b"event: message_stop\n"
        b"data: {}\n"
        b"\n"
    )
    assert parse(raw) == parse(*[raw[i : i + 1] for i in range(len(raw))])


def test_crlf_and_lone_cr_terminators():
    assert parse(b"event: a\r\ndata: 1\r\n\r\n")[0].data == "1"
    assert parse(b"event: b\rdata: 2\r\r")[0].data == "2"


def test_crlf_split_across_chunks_is_not_a_blank_line():
    """A trailing \\r must not be consumed until we know what follows it."""
    frames = parse(b"event: a\r", b"\ndata: 1\r\n\r\n")
    assert frames == [SSEFrame(event="a", data="1")]


def test_utf8_split_across_chunks():
    """A multi-byte character straddling a chunk boundary must survive."""
    payload = '{"text":"→ 日本語"}'.encode()
    raw = b"event: d\ndata: " + payload + b"\n\n"
    mid = len(b"event: d\ndata: ") + 17  # lands inside a multi-byte sequence
    frames = parse(raw[:mid], raw[mid:])
    assert frames[0].data == '{"text":"→ 日本語"}'


def test_one_leading_space_is_stripped_but_not_two():
    assert parse(b"data:  padded\n\n")[0].data == " padded"


def test_incomplete_trailing_frame_is_discarded():
    assert parse(b"event: a\ndata: 1\n") == []