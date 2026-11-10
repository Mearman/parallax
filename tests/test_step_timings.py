from parallax.server.speculation import StepTimings


def test_summary_averages_each_phase_and_forgets_the_request():
    timings = StepTimings()
    timings.record("r", "round_trip", 0.1)
    timings.record("r", "round_trip", 0.3)
    timings.record("r", "m3_chunk", 0.05)
    line = timings.summary("r")
    assert "round_trip 200.0 ms over 2" in line
    assert "m3_chunk 50.0 ms over 1" in line
    assert timings.summary("r") == ""


def test_summary_of_an_unrecorded_request_is_empty():
    assert StepTimings().summary("missing") == ""


def test_speculation_summary_counts_steps_by_drafted_slots_and_accepted_drafts():
    from parallax.server.speculation import SpeculationStats

    stats = SpeculationStats()
    stats.record("r", 3, 4)
    stats.record("r", 3, 1)
    stats.record("r", 3, 1)
    stats.record("r", 2, 2)
    line = stats.summary("r")
    assert "3 slots: 0 accepted in 2, 3 accepted in 1" in line
    assert "2 slots: 1 accepted in 1" in line
    assert "4 of 11 drafted slots accepted" in line
