from app.ml.failure_parser import parse


def test_parse_python_failure_extracts_root_frame_and_diff() -> None:
    raw_log = """AssertionError: Expected 200, got 502
  File \"/workspace/tests/test_checkout.py\", line 42, in test_payment_gateway
  File \"/usr/lib/python3.12/site-packages/_pytest/python.py\", line 157, in pytest_pyfunc_call
AssertionError: Expected 200, got 502
"""

    parsed = parse(raw_log)

    assert parsed.exception_type == "AssertionError"
    assert parsed.message == "Expected 200, got 502"
    assert parsed.root_file == "test_checkout.py"
    assert parsed.root_line == 42
    assert parsed.expected == "200"
    assert parsed.actual == "got 502"


def test_parse_java_failure_extracts_root_frame() -> None:
    raw_log = """java.lang.IllegalStateException: state mismatch
    at com.myco.checkout.PaymentService.charge(PaymentService.java:88)
    at org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:55)
"""

    parsed = parse(raw_log)

    assert parsed.exception_type == "java.lang.IllegalStateException"
    assert parsed.root_file == "PaymentService.java"
    assert parsed.root_line == 88