"""Run the terminal as a background Windows task.

The trading loop is already a long-running process; this only changes *who
starts it and what happens when it dies*. No trading code is touched.

## Why a scheduled task rather than a true NT service

A Windows service must implement the Service Control Manager protocol -- it has
to answer a start request within a timeout and respond to control codes.
Registering an ordinary executable with ``sc create`` produces a service that
Windows kills seconds after starting it, reporting only that it "did not
respond to the start request in a timely fashion". Making this a real service
would mean a pywin32 service wrapper bundled into the frozen executable, which
is fragile under PyInstaller and buys nothing here.

A scheduled task started at logon gives the same practical result -- automatic
start, restart on failure, survives closing the window -- using only what
Windows ships with.

## The account question, which is the real constraint

Exchange credentials live in ``%USERPROFILE%\\.godalgo``. That directory is
readable by the user who owns it and, since the Windows ACL fix, by nobody
else. Two consequences follow, and they are the whole reason the default is
what it is:

* Running **as SYSTEM at boot** starts without anyone logged in -- and cannot
  read those credentials at all. It would come up and trade nothing.
* Running **as you at boot** needs your password stored in Task Scheduler.

So the default trigger is **at logon**: no password, full access to your
profile, survives closing the terminal window, and comes back automatically
after a reboot once you sign in. ``--at-boot`` is available for the password
route, and says so plainly.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape as _escape

__all__ = ["TASK_NAME", "install", "main", "start", "status", "stop", "uninstall"]

TASK_NAME = "GODALGO Terminal"


def _require_windows() -> None:
    if sys.platform != "win32":
        raise RuntimeError(
            "the scheduled-task installer is Windows-only; on Linux or macOS "
            "use systemd or launchd"
        )


def _executable() -> tuple[str, list[str]]:
    """The command and leading arguments that start the terminal.

    Task Scheduler keeps the program and its arguments in separate elements, so
    these are returned separately rather than as one shell string -- an
    ``<Command>`` holding ``"python" "script.py"`` is not parsed, it is looked
    up as a file with that literal name.
    """
    if getattr(sys, "frozen", False):
        return sys.executable, []
    # Source checkout: re-invoke this interpreter against the launcher.
    launcher = Path(__file__).resolve().parents[2] / "run-terminal.py"
    return sys.executable, [str(launcher)]


def _task_xml(
    command: str, prefix: list[str], *, at_boot: bool, port: int
) -> str:
    """Build the task definition.

    Written as XML rather than assembled from ``schtasks`` flags because the
    settings that actually matter for a 24/7 process -- no execution time
    limit, restart on failure, and above all a single-instance policy -- are
    not reachable from the command-line form.
    """
    trigger = (
        "<BootTrigger><Enabled>true</Enabled></BootTrigger>"
        if at_boot
        else "<LogonTrigger><Enabled>true</Enabled></LogonTrigger>"
    )
    logon_type = "Password" if at_boot else "InteractiveToken"
    user = _escape(
        f"{os.environ.get('USERDOMAIN', '')}\\{getpass.getuser()}".lstrip("\\")
    )
    arguments = _escape(
        " ".join([*(f'"{p}"' for p in prefix), "--no-browser", "--port", str(port)])
    )

    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>GODALGO trading terminal (port {port})</Description>
  </RegistrationInfo>
  <Triggers>{trigger}</Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{user}</UserId>
      <LogonType>{logon_type}</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>5</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{_escape(command)}</Command>
      <Arguments>{arguments}</Arguments>
    </Exec>
  </Actions>
</Task>
"""


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def install(port: int = 8787, *, at_boot: bool = False) -> int:
    """Register the task. Returns a process exit code."""
    _require_windows()

    command, prefix = _executable()
    xml = _task_xml(command, prefix, at_boot=at_boot, port=port)

    # Task Scheduler requires UTF-16 with a BOM; a UTF-8 file is rejected with
    # an error that does not mention encoding.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, encoding="utf-16",
    ) as handle:
        handle.write(xml)
        path = handle.name

    try:
        args = ["schtasks", "/Create", "/TN", TASK_NAME, "/XML", path, "/F"]
        if at_boot:
            # Boot start runs with no interactive session, so Task Scheduler
            # needs the password to log the account on.
            print("Boot start runs as your account with no session, so Windows")
            print("needs your password to start it. It is stored by Task")
            print("Scheduler, not by this program.")
            password = getpass.getpass("Windows password: ")
            args += ["/RU", getpass.getuser(), "/RP", password]

        result = _run(args)
        if result.returncode != 0:
            print(f"could not register the task: {result.stderr.strip()}", file=sys.stderr)
            return result.returncode
    finally:
        Path(path).unlink(missing_ok=True)

    when = "at boot" if at_boot else "when you sign in"
    print(f"registered '{TASK_NAME}' — starts {when}, restarts if it stops")
    print(f"  terminal:  http://127.0.0.1:{port}")
    print("  start now: godalgo-terminal.exe service start")
    print("  remove:    godalgo-terminal.exe service uninstall")
    if not at_boot:
        print("\nNote: it starts when you sign in, not before. Running at boot")
        print("without a session would mean running as SYSTEM, which cannot read")
        print("the exchange credentials in your user profile.")
    return 0


def uninstall() -> int:
    _require_windows()
    result = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if result.returncode != 0:
        print(f"could not remove the task: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"removed '{TASK_NAME}'")
    return 0


def start() -> int:
    _require_windows()
    result = _run(["schtasks", "/Run", "/TN", TASK_NAME])
    if result.returncode != 0:
        print(f"could not start: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"started '{TASK_NAME}'")
    return 0


def stop() -> int:
    """Stop the task.

    This terminates the process. It does not flatten positions -- the engine's
    own shutdown path does that, and a task stop does not go through it. Close
    the bot from the terminal if you want an orderly exit.
    """
    _require_windows()
    result = _run(["schtasks", "/End", "/TN", TASK_NAME])
    if result.returncode != 0:
        print(f"could not stop: {result.stderr.strip()}", file=sys.stderr)
        return result.returncode
    print(f"stopped '{TASK_NAME}'")
    print("note: this kills the process; it does not close open positions")
    return 0


def status() -> int:
    _require_windows()
    result = _run(["schtasks", "/Query", "/TN", TASK_NAME, "/V", "/FO", "LIST"])
    if result.returncode != 0:
        print(f"'{TASK_NAME}' is not registered")
        return 1
    wanted = ("Status:", "Last Run Time:", "Last Result:", "Next Run Time:",
              "Scheduled Task State:", "Task To Run:")
    for line in result.stdout.splitlines():
        if any(line.strip().startswith(k) for k in wanted):
            print(f"  {line.strip()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """``godalgo-terminal service <command>``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="godalgo-terminal service",
        description="run the terminal in the background on Windows",
    )
    parser.add_argument(
        "command", choices=["install", "uninstall", "start", "stop", "status"],
    )
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--at-boot", action="store_true",
        help="start at boot instead of at sign-in; prompts for your Windows "
             "password, which Task Scheduler needs to start a session-less task",
    )
    args = parser.parse_args(argv)

    try:
        if args.command == "install":
            return install(port=args.port, at_boot=args.at_boot)
        return {"uninstall": uninstall, "start": start,
                "stop": stop, "status": status}[args.command]()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
