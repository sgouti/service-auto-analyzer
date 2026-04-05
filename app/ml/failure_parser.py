from __future__ import annotations

import re
from dataclasses import dataclass

FRAMEWORK_PACKAGES = {
    "pytest",
    "unittest",
    "_pytest",
    "pluggy",
    "java.lang.reflect",
    "sun.reflect",
    "org.junit",
    "com.google.common",
}

EXCEPTION_PATTERN = re.compile(r"^(\w[\w.]*(?:Error|Exception|Failure)): (.+)$")
PYTHON_FRAME_PATTERN = re.compile(r'\s*File "(.+)", line (\d+)')
JAVA_FRAME_PATTERN = re.compile(r"\s*at ([\w.$]+)\(([^:]+):(\d+)\)")
EXPECTED_ACTUAL_PATTERN = re.compile(r"[Ee]xpected (.+?)(?:,| but was) (.+)")


@dataclass(frozen=True)
class ParsedFailure:
    exception_type: str
    message: str
    root_file: str
    root_line: int
    expected: str | None
    actual: str | None
    raw_log: str


def parse(raw_log: str) -> ParsedFailure:
    lines = [line.rstrip() for line in (raw_log or "").splitlines() if line.strip()]
    exception_type, message = _extract_exception(lines)
    root_file, root_line = _find_root_frame(lines)
    expected, actual = _extract_diff(message)
    return ParsedFailure(
        exception_type=exception_type,
        message=message,
        root_file=root_file,
        root_line=root_line,
        expected=expected,
        actual=actual,
        raw_log=raw_log,
    )


def _extract_exception(lines: list[str]) -> tuple[str, str]:
    for line in reversed(lines):
        match = EXCEPTION_PATTERN.match(line)
        if match:
            return match.group(1), match.group(2)
    return "UnknownError", (lines[-1] if lines else "")


def _find_root_frame(lines: list[str]) -> tuple[str, int]:
    for line in lines:
        python_match = PYTHON_FRAME_PATTERN.match(line)
        if python_match:
            file_path = python_match.group(1)
            if not any(package in file_path for package in FRAMEWORK_PACKAGES):
                return file_path.split("/")[-1], int(python_match.group(2))

        java_match = JAVA_FRAME_PATTERN.match(line)
        if java_match:
            class_name, file_name, line_num = java_match.groups()
            if not any(package in class_name for package in FRAMEWORK_PACKAGES):
                return file_name, int(line_num)
    return "unknown", 0


def _extract_diff(message: str) -> tuple[str | None, str | None]:
    match = EXPECTED_ACTUAL_PATTERN.search(message or "")
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()