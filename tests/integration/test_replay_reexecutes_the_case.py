# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Metacognition AI
#
# This source code is licensed under the AGPL-3.0-only licence found in the
# LICENSE file in the root directory of this source tree.

"""``zeos replay --case`` runs the tree again and holds the run against the journal.

Re-reading a journal proves the file round-trips. The claim worth checking is the
other one: that the same tree, schedule and seed write the same journal again. A
journal names its tree by digest, so the check can refuse a tree that has been edited
since rather than report a divergence that says nothing about the kernel.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from zeos.cli import main
from zeos.core.events import KernelStarted, PipeWritten
from zeos.core.kernel import Kernel, KernelConfig
from zeos.core.pipes import PipeTable
from zeos.core.vectors import VectorTable
from zeos.descriptor.loader import case_digest, load_case
from zeos.journal.writer import Journal, read_journal
from zeos.machine.scripted import ScriptedMachine
from zeos.world.store import WorldStore

SMOKE = Path(__file__).parent.parent / "fixtures" / "smoke"


def _tree(tmp_path: Path, name: str = "smoke") -> Path:
    return Path(shutil.copytree(SMOKE, tmp_path / name))


def _record(case: Path, journal: Path, *machine: str) -> None:
    run = ["run", str(case), "--events", str(case / "events.jsonl"), "--journal", str(journal)]
    assert main([*run, "--seed", "7", "--block-size", "8", "--quiet", *machine]) == 0


def _replay(case: Path, journal: Path, *extra: str) -> int:
    return main(["replay", str(journal), "--case", str(case), *extra])


@pytest.mark.determinism
@pytest.mark.parametrize("machine", ["scripted", "seat"])
def test_a_recorded_run_reexecutes_byte_identically(tmp_path: Path, machine: str) -> None:
    case, journal = _tree(tmp_path), tmp_path / "run.jsonl"
    _record(case, journal, "--machine", machine)
    events = ("--events", str(case / "events.jsonl"), "--machine", machine)
    assert _replay(case, journal, *events) == 0


def test_the_journal_names_the_tree_it_is_a_run_of(tmp_path: Path) -> None:
    case, journal = _tree(tmp_path), tmp_path / "run.jsonl"
    _record(case, journal)
    started = read_journal(journal)[0].event
    assert isinstance(started, KernelStarted)
    assert started.case_digest == case_digest(case) == load_case(case).digest


@pytest.mark.determinism
def test_a_tree_that_was_moved_is_still_the_same_tree(tmp_path: Path) -> None:
    case, journal = _tree(tmp_path), tmp_path / "run.jsonl"
    _record(case, journal)
    moved = case.rename(tmp_path / "elsewhere")
    assert _replay(moved, journal, "--events", str(moved / "events.jsonl")) == 0


def test_a_tree_edited_since_the_run_is_refused_not_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case, journal = _tree(tmp_path), tmp_path / "run.jsonl"
    _record(case, journal)
    goal = case / "goals" / "supervision.md"
    goal.write_text(goal.read_text("utf-8") + "\nOne more instruction.\n", "utf-8")

    assert _replay(case, journal, "--events", str(case / "events.jsonl")) == 1
    err = capsys.readouterr().err
    assert "is not the tree this journal is a run of" in err
    assert "DIVERGES" not in err, "refused before running, so nothing diverged"


def test_a_missing_delivery_is_reported_at_the_event_it_would_have_been(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case, journal = _tree(tmp_path), tmp_path / "run.jsonl"
    _record(case, journal)
    delivery = next(
        r.seq
        for r in read_journal(journal)
        if isinstance(r.event, PipeWritten) and r.event.job is None
    )

    assert _replay(case, journal) == 1, "no schedule given, so the alarm never arrives"
    assert f"DIVERGES at seq {delivery} " in capsys.readouterr().err


def test_a_journal_that_names_no_tree_cannot_be_held_against_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    kernel = Kernel(
        descriptors={},
        machine=ScriptedMachine({}, block_size=8),
        pipes=PipeTable(()),
        vectors=VectorTable(),
        world=WorldStore(),
        journal_sink=[],
        config=KernelConfig(case="assembled-in-code"),
    )
    kernel.start()
    with Journal(tmp_path / "run.jsonl") as journal:
        journal.extend(kernel.events)

    assert _replay(SMOKE, tmp_path / "run.jsonl") == 1
    assert "names no case tree" in capsys.readouterr().err


def test_the_digest_follows_content_and_names_but_not_location(tmp_path: Path) -> None:
    here, there = _tree(tmp_path, "here"), _tree(tmp_path, "there")
    assert case_digest(here) == case_digest(there)

    pipes = there / "system" / "pipes.yaml"
    pipes.write_bytes(pipes.read_bytes() + b"\n")
    assert case_digest(here) != case_digest(there), "one byte of configuration"

    goal = here / "goals" / "supervision.md"
    before = case_digest(here)
    goal.rename(goal.with_name("oversight.md"))
    assert case_digest(here) != before, "a rename is a change"


def test_the_schedule_is_an_input_to_a_run_not_part_of_the_tree(tmp_path: Path) -> None:
    case = _tree(tmp_path)
    before = case_digest(case)
    (case / "events.jsonl").write_text("", "utf-8")
    assert case_digest(case) == before
