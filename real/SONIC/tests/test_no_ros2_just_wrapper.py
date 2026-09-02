import os
import platform
import subprocess
from pathlib import Path


WRAPPER = Path(__file__).parents[1] / "scripts" / "no_ros2_bin" / "just"


def test_wrapper_prioritizes_unitree_dds_over_ros(tmp_path):
    dds_lib = (
        tmp_path
        / "thirdparty"
        / "unitree_sdk2"
        / "thirdparty"
        / "lib"
        / platform.machine()
    )
    dds_lib.mkdir(parents=True)

    fake_just = tmp_path / "fake-just"
    fake_just.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'HAS_ROS2=%s\\n' \"$HAS_ROS2\"\n"
        "printf 'LD_LIBRARY_PATH=%s\\n' \"$LD_LIBRARY_PATH\"\n"
    )
    fake_just.chmod(0o755)

    env = os.environ.copy()
    env.update(
        REAL_JUST=str(fake_just),
        LD_LIBRARY_PATH="/opt/ros/humble/lib/x86_64-linux-gnu",
    )
    result = subprocess.run(
        [str(WRAPPER), "run", "g1_deploy_onnx_ref"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert values["HAS_ROS2"] == "0"
    assert values["LD_LIBRARY_PATH"].split(":", 1)[0] == str(dds_lib)
