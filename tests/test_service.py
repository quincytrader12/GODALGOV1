"""The background-task installer.

Only the parts that can be checked off Windows are covered: the task
definition, and the refusal to touch anything on other platforms. The
``schtasks`` calls themselves are one-line wrappers whose only interesting
behaviour is passing the XML through, which is what the definition tests are
for.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import pytest

from godalgo import service

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def _xml(**kwargs) -> ET.Element:
    kwargs.setdefault("at_boot", False)
    kwargs.setdefault("port", 8787)
    text = service._task_xml(r"C:\godalgo\godalgo-terminal.exe", [], **kwargs)
    # The declaration says UTF-16 because Task Scheduler requires the file to
    # be encoded that way; strip it so ElementTree will parse the str.
    return ET.fromstring(text.split("?>", 1)[1])


def _text(root: ET.Element, path: str) -> str:
    node = root.find(path, NS)
    assert node is not None, f"missing {path}"
    return (node.text or "").strip()


def test_task_definition_is_valid_xml_and_runs_the_terminal():
    root = _xml()
    assert _text(root, ".//t:Actions/t:Exec/t:Command").endswith("godalgo-terminal.exe")
    arguments = _text(root, ".//t:Actions/t:Exec/t:Arguments")
    assert "--port 8787" in arguments
    # Without this the task opens a browser tab on a machine nobody is looking
    # at, once per restart.
    assert "--no-browser" in arguments


def test_a_restart_cannot_start_a_second_bot():
    """The single most important setting in the file.

    Two instances would run two independent strategy stacks against the same
    account: both size off the same buying power, and neither knows the other's
    position. ``IgnoreNew`` is what makes a restart safe.
    """
    assert _text(_xml(), ".//t:MultipleInstancesPolicy") == "IgnoreNew"


def test_the_task_is_not_killed_by_time_or_battery():
    root = _xml()
    # PT0S means no limit. The default is 72 hours, after which Windows would
    # terminate a running bot mid-position without warning.
    assert _text(root, ".//t:ExecutionTimeLimit") == "PT0S"
    assert _text(root, ".//t:StopIfGoingOnBatteries") == "false"
    assert _text(root, ".//t:DisallowStartIfOnBatteries") == "false"
    assert _text(root, ".//t:RunOnlyIfIdle") == "false"
    assert _text(root, ".//t:IdleSettings/t:StopOnIdleEnd") == "false"


def test_it_comes_back_after_a_crash():
    root = _xml()
    assert _text(root, ".//t:RestartOnFailure/t:Interval") == "PT1M"
    assert int(_text(root, ".//t:RestartOnFailure/t:Count")) >= 100


def test_default_trigger_is_logon_so_credentials_are_readable():
    """At logon, with an interactive token, deliberately.

    The exchange keys live in the user's profile and are ACL-restricted to that
    user. A boot trigger under SYSTEM would start on time and then be unable to
    read them.
    """
    root = _xml()
    assert root.find(".//t:LogonTrigger", NS) is not None
    assert root.find(".//t:BootTrigger", NS) is None
    assert _text(root, ".//t:Principal/t:LogonType") == "InteractiveToken"


def test_boot_trigger_switches_to_a_stored_password():
    root = _xml(at_boot=True)
    assert root.find(".//t:BootTrigger", NS) is not None
    assert _text(root, ".//t:Principal/t:LogonType") == "Password"


def test_port_reaches_both_the_arguments_and_the_description():
    root = _xml(port=9001)
    assert "--port 9001" in _text(root, ".//t:Actions/t:Exec/t:Arguments")
    assert "9001" in _text(root, ".//t:RegistrationInfo/t:Description")


def test_an_ampersand_in_a_path_does_not_break_the_file():
    """Paths like ``C:\\Users\\R&D\\...`` are legal on Windows.

    Interpolating one raw would produce a file Task Scheduler rejects with a
    parse error that names neither the character nor the path.
    """
    text = service._task_xml(r"C:\R&D\godalgo.exe", [], at_boot=False, port=8787)
    root = ET.fromstring(text.split("?>", 1)[1])
    assert _text(root, ".//t:Actions/t:Exec/t:Command") == r"C:\R&D\godalgo.exe"


def test_source_checkout_passes_the_launcher_as_an_argument():
    """Not as part of the command.

    ``<Command>`` is a path, not a shell line: an interpreter and a script
    concatenated into it would be looked up as a single file of that name.
    """
    command, prefix = service._executable()
    assert command == sys.executable
    assert prefix and prefix[0].endswith("run-terminal.py")

    text = service._task_xml(command, prefix, at_boot=False, port=8787)
    root = ET.fromstring(text.split("?>", 1)[1])
    assert _text(root, ".//t:Actions/t:Exec/t:Command") == sys.executable
    assert "run-terminal.py" in _text(root, ".//t:Actions/t:Exec/t:Arguments")


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows refusal")
@pytest.mark.parametrize(
    "action", [service.install, service.uninstall, service.start,
               service.stop, service.status],
)
def test_every_action_refuses_off_windows(action):
    with pytest.raises(RuntimeError, match="Windows-only"):
        action()


@pytest.mark.skipif(sys.platform == "win32", reason="checks the non-Windows refusal")
def test_cli_reports_the_refusal_instead_of_raising(capsys):
    assert service.main(["status"]) == 2
    assert "Windows-only" in capsys.readouterr().err
