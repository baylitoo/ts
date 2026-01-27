import pytest
from rl_money_laundering.hello import main


def test_main_output(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    captured = capsys.readouterr()
    assert "Hello from rl-money-laundering!" in captured.out
