"""The Phase 4, 50-case safety evaluation for exact file edits."""

from dataclasses import dataclass
from pathlib import Path

from agent.tools.edit_file import EditFileTool
from agent.tools.workspace import Workspace


@dataclass(frozen=True)
class EditCase:
    name: str
    arguments: dict[str, object]
    files: tuple[tuple[str, bytes], ...] = ()
    directories: tuple[str, ...] = ()
    expected_files: tuple[tuple[str, bytes], ...] = ()
    expected_absent: tuple[str, ...] = ()
    expect_error: bool = False
    expected_message: str | None = None
    max_bytes: int = 1_000_000


def _success(
    name: str,
    path: str,
    source: str,
    old_text: str,
    new_text: str,
    *,
    max_bytes: int = 1_000_000,
) -> EditCase:
    return EditCase(
        name=name,
        arguments={
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
        },
        files=((path, source.encode("utf-8")),),
        expected_files=(
            (path, source.replace(old_text, new_text, 1).encode("utf-8")),
        ),
        max_bytes=max_bytes,
    )


def _refusal(
    name: str,
    path: str,
    source: str,
    old_text: str,
    new_text: str,
    message: str,
    *,
    max_bytes: int = 1_000_000,
) -> EditCase:
    return EditCase(
        name=name,
        arguments={
            "path": path,
            "old_text": old_text,
            "new_text": new_text,
        },
        files=((path, source.encode("utf-8")),),
        expected_files=((path, source.encode("utf-8")),),
        expect_error=True,
        expected_message=message,
        max_bytes=max_bytes,
    )


CASES = (
    # Expected successful edits: each must leave exactly the intended file.
    _success("01-simple", "notes.txt", "hello NAME", "NAME", "Sanjay"),
    _success("02-middle", "notes.txt", "before OLD after", "OLD", "NEW"),
    _success("03-at-start", "notes.txt", "OLD then text", "OLD", "NEW"),
    _success("04-at-end", "notes.txt", "text then OLD", "OLD", "NEW"),
    _success("05-multiline-block", "notes.txt", "a\nOLD\nblock\nz", "OLD\nblock", "NEW"),
    _success("06-blank-line", "notes.txt", "top\n\nOLD\n", "\nOLD", "\nNEW"),
    _success("07-delete", "notes.txt", "keep REMOVE keep", "REMOVE ", ""),
    _success("08-unicode", "notes.txt", "café OLD", "OLD", "nouveau"),
    _success("09-emoji", "notes.txt", "status: OLD", "OLD", "✅"),
    _success("10-identical", "notes.txt", "only token", "token", "token"),
    _success("11-size-boundary", "notes.txt", "one", "one", "two", max_bytes=3),
    _success("12-add-lines", "notes.txt", "header OLD footer", "OLD", "one\ntwo"),
    _success("13-nested-file", "nested/settings.txt", "mode=OLD", "OLD", "safe"),
    _success("14-literal-regex-chars", "notes.txt", "value=a+b", "a+b", "sum"),
    _success("15-preserve-spaces", "notes.txt", "  OLD  ", "OLD", "new"),
    _success("16-crlf", "notes.txt", "before\r\nOLD\r\nafter\r\n", "OLD", "NEW"),
    _success("17-large-text", "notes.txt", "x" * 900 + " OLD", "OLD", "NEW"),
    _success("18-near-end", "notes.txt", "prefix " + "x" * 100 + " OLD", "OLD", "NEW"),
    _success("19-newline-replacement", "notes.txt", "value=OLD", "OLD", "first\nsecond"),
    _success("20-dotted-path", "./notes.txt", "OLD", "OLD", "NEW"),
    _success("21-spaced-filename", "release notes.txt", "OLD", "OLD", "NEW"),
    _success("22-full-file", "notes.txt", "OLD", "OLD", "entire replacement"),
    _success("23-prefix-substring", "notes.txt", "foobar", "foo", "bar"),
    _success("24-accented-match", "notes.txt", "replace é here", "é", "e"),
    _success("25-tabbed-text", "notes.txt", "key\tOLD\tvalue", "OLD", "NEW"),
    # Expected refusals: each must preserve the original file or workspace.
    _refusal("26-missing-match", "notes.txt", "original", "missing", "new", "old_text was not found in the file"),
    _refusal("27-two-matches", "notes.txt", "OLD and OLD", "OLD", "NEW", "old_text occurs 2 times; provide a unique match"),
    _refusal("28-three-matches", "notes.txt", "OLD OLD OLD", "OLD", "NEW", "old_text occurs 3 times; provide a unique match"),
    _refusal("29-overlapping-matches", "notes.txt", "aaa", "aa", "b", "old_text occurs 2 times; provide a unique match"),
    EditCase(
        name="30-empty-old-text",
        arguments={"path": "notes.txt", "old_text": "", "new_text": "new"},
        files=(("notes.txt", b"original"),),
        expected_files=(("notes.txt", b"original"),),
        expect_error=True,
        expected_message="old_text must not be empty",
    ),
    EditCase(
        name="31-missing-file",
        arguments={"path": "missing.txt", "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="file does not exist: missing.txt",
    ),
    EditCase(
        name="32-directory-target",
        arguments={"path": "folder", "old_text": "old", "new_text": "new"},
        directories=("folder",),
        expect_error=True,
        expected_message="path is not a regular file: folder",
    ),
    EditCase(
        name="33-binary-file",
        arguments={"path": "notes.bin", "old_text": "old", "new_text": "new"},
        files=(("notes.bin", b"old\x00data"),),
        expected_files=(("notes.bin", b"old\x00data"),),
        expect_error=True,
        expected_message="binary files cannot be edited as text",
    ),
    EditCase(
        name="34-non-utf8-file",
        arguments={"path": "notes.bin", "old_text": "old", "new_text": "new"},
        files=(("notes.bin", b"\xff\xfe"),),
        expected_files=(("notes.bin", b"\xff\xfe"),),
        expect_error=True,
        expected_message="file is not valid UTF-8 text",
    ),
    EditCase(
        name="35-path-traversal",
        arguments={"path": "../outside.txt", "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="path escapes the workspace",
    ),
    EditCase(
        name="36-absolute-path",
        arguments={"path": "/tmp/outside.txt", "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="absolute paths are not allowed",
    ),
    EditCase(
        name="37-null-byte-path",
        arguments={"path": "bad\x00path", "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="path must not contain a null byte",
    ),
    EditCase(
        name="38-unknown-argument",
        arguments={"path": "notes.txt", "old_text": "old", "new_text": "new", "extra": True},
        files=(("notes.txt", b"old"),),
        expected_files=(("notes.txt", b"old"),),
        expect_error=True,
        expected_message="unexpected arguments: extra",
    ),
    EditCase(
        name="39-old-text-wrong-type",
        arguments={"path": "notes.txt", "old_text": 1, "new_text": "new"},
        files=(("notes.txt", b"old"),),
        expected_files=(("notes.txt", b"old"),),
        expect_error=True,
        expected_message="old_text must be a string",
    ),
    EditCase(
        name="40-new-text-wrong-type",
        arguments={"path": "notes.txt", "old_text": "old", "new_text": None},
        files=(("notes.txt", b"old"),),
        expected_files=(("notes.txt", b"old"),),
        expect_error=True,
        expected_message="new_text must be a string",
    ),
    EditCase(
        name="41-path-wrong-type",
        arguments={"path": None, "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="path must be a string",
    ),
    _refusal("42-result-too-large", "notes.txt", "one", "one", "longer", "edited file exceeds the 4-byte limit", max_bytes=4),
    _refusal("43-source-too-large", "notes.txt", "large", "large", "small", "file exceeds the 4-byte edit limit", max_bytes=4),
    _refusal("44-no-match-empty-file", "notes.txt", "", "old", "new", "old_text was not found in the file"),
    _refusal("45-duplicate-across-lines", "notes.txt", "OLD\nOLD\n", "OLD", "NEW", "old_text occurs 2 times; provide a unique match"),
    _refusal("46-duplicate-multiline-block", "notes.txt", "a\nb\na\nb\n", "a\nb", "new", "old_text occurs 2 times; provide a unique match"),
    EditCase(
        name="47-old-text-boolean",
        arguments={"path": "notes.txt", "old_text": True, "new_text": "new"},
        files=(("notes.txt", b"old"),),
        expected_files=(("notes.txt", b"old"),),
        expect_error=True,
        expected_message="old_text must be a string",
    ),
    EditCase(
        name="48-new-text-boolean",
        arguments={"path": "notes.txt", "old_text": "old", "new_text": False},
        files=(("notes.txt", b"old"),),
        expected_files=(("notes.txt", b"old"),),
        expect_error=True,
        expected_message="new_text must be a string",
    ),
    EditCase(
        name="49-empty-path",
        arguments={"path": "", "old_text": "old", "new_text": "new"},
        expect_error=True,
        expected_message="path must not be empty",
    ),
    _refusal("50-line-ending-does-not-match", "notes.txt", "old\r\n", "old\n", "new\n", "old_text was not found in the file"),
)


async def _run_case(case: EditCase, root: Path) -> tuple[bool, str]:
    case_root = root / case.name
    case_root.mkdir()
    for directory in case.directories:
        (case_root / directory).mkdir(parents=True)
    for relative_path, content in case.files:
        path = case_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    tool = EditFileTool(Workspace(case_root), max_bytes=case.max_bytes)
    result = await tool.execute(case.arguments)
    passed = result.is_error is case.expect_error

    if case.expected_message is not None:
        passed = passed and result.content == case.expected_message
    for relative_path, expected_content in case.expected_files:
        path = case_root / relative_path
        passed = passed and path.is_file() and path.read_bytes() == expected_content
    for relative_path in case.expected_absent:
        passed = passed and not (case_root / relative_path).exists()

    diagnostic = (
        f"{case.name}: is_error={result.is_error}, content={result.content!r}"
    )
    return passed, diagnostic


async def test_edit_file_passes_the_phase_4_fifty_case_evaluation(
    tmp_path: Path,
):
    assert len(CASES) == 50

    outcomes = [await _run_case(case, tmp_path) for case in CASES]
    failures = [diagnostic for passed, diagnostic in outcomes if not passed]
    failure_rate = len(failures) / len(CASES)

    assert failure_rate < 0.05, (
        f"edit-application failure rate: {failure_rate:.0%}; "
        f"failures:\n" + "\n".join(failures)
    )
