from __future__ import annotations

from agi_core_codex.minimal.core import (
    CandidateResult,
    CandidateScore,
    GridExample,
    GridTask,
    LearnerMemory,
    WakeSleepLearner,
    _program_rank_key,
    compose_program,
    make_leaf_program,
)
from agi_core_codex.minimal.domains import build_synthetic_curriculum
from agi_core_codex.minimal.ops import compositor_specs, unary_seed_specs


def test_learner_memory_stages_before_commit() -> None:
    leaf = make_leaf_program(unary_seed_specs()[0])
    memory = LearnerMemory()
    assert memory.committed_programs() == ()

    from agi_core_codex.minimal.core import LibraryEntry

    memory.stage(LibraryEntry(program=leaf, promotion_score=(1.0, 1, 0), promoted_round=0, source_task_keys=("a",)))
    assert memory.committed_programs() == ()
    assert memory.commit() == 1
    assert memory.committed_programs() == (leaf,)


def test_program_ids_are_stable_for_same_ast() -> None:
    crop = make_leaf_program(next(spec for spec in unary_seed_specs() if spec.name == "crop_support"))
    flip = make_leaf_program(next(spec for spec in unary_seed_specs() if spec.name == "flip_h"))
    chain = next(spec for spec in compositor_specs() if spec.name == "chain")

    first = compose_program(chain, crop, flip)
    second = compose_program(chain, crop, flip)

    assert first.id == second.id
    assert first.name == second.name


def test_candidate_ranking_prefers_accuracy_then_simplicity_then_failures() -> None:
    identity = make_leaf_program(unary_seed_specs()[0])
    simple = CandidateResult(identity, CandidateScore(False, 0.9, 0), 1, False)
    complex_program = compose_program(next(spec for spec in compositor_specs() if spec.name == "chain"), identity, identity)
    complex_candidate = CandidateResult(complex_program, CandidateScore(False, 0.9, 0), 1, False)
    failing = CandidateResult(identity, CandidateScore(False, 0.9, 1), 1, False)

    assert _program_rank_key(simple) > _program_rank_key(complex_candidate)
    assert _program_rank_key(simple) > _program_rank_key(failing)


def test_sleep_promotes_reused_subprograms() -> None:
    learner = WakeSleepLearner(unary_primitives=unary_seed_specs(), binary_compositors=compositor_specs())
    memory = LearnerMemory()
    task = GridTask(
        task_id="crop_task",
        train=(GridExample(input=((0, 1, 0),), output=((1,),)),),
        test=(),
    )
    task_runs, _ = learner.run_round(tasks=(task,), memory=memory, round_index=0)
    assert task_runs[0].best is not None
    promoted = learner.sleep(task_runs=task_runs, memory=memory, round_index=0)
    assert promoted >= 0


def test_promoted_compound_programs_are_canonicalized() -> None:
    learner = WakeSleepLearner(unary_primitives=unary_seed_specs(), binary_compositors=compositor_specs())
    memory = LearnerMemory()
    tasks = build_synthetic_curriculum("pair").tasks
    learner.run_round(tasks=tasks, memory=memory, round_index=0)
    learner.run_round(tasks=tasks, memory=memory, round_index=1)

    committed_names = {entry.program.name for entry in memory.committed.values()}
    assert "crop_support-chain-flip_h" in committed_names
    assert all(not name.startswith("lib:") for name in committed_names)
    assert memory.reused_program_count() > 0
    assert memory.total_reuse_count() > 0


def test_later_round_search_filters_redundant_primitive_library_entries() -> None:
    learner = WakeSleepLearner(unary_primitives=unary_seed_specs(), binary_compositors=compositor_specs())
    memory = LearnerMemory()
    tasks = build_synthetic_curriculum("pair").tasks
    learner.run_round(tasks=tasks, memory=memory, round_index=0)

    active_entries = learner._search_library_entries(memory, round_index=1)
    active_names = {entry.program.name for entry in active_entries}

    assert active_names == {"crop_support"}
