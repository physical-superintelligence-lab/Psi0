from types import SimpleNamespace

from real.SONIC.run_sim_loop_single_dds import disable_outer_channel_init


def test_outer_channel_initialization_is_suppressed():
    calls = []
    run_sim_loop = SimpleNamespace(init_channel=lambda config: calls.append(config))

    disable_outer_channel_init(run_sim_loop)
    run_sim_loop.init_channel(config={"DOMAIN_ID": 0, "INTERFACE": "lo"})

    assert calls == []
