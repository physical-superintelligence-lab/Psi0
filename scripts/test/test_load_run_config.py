from pathlib import Path
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything
from psi.config.config import LaunchConfig

if __name__ == "__main__":

    run_dir = Path(".runs/psix_finetune/psix_subtask.sonic_psix_g1_mix.flow1000.cosine.lr5.0e-05.b16.gpus1.2606090222")
    # build dynamic config, then load the saved json
    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = (run_dir / "run_config.json").open("r").read()
    launch_config = config_.model_validate_json(conf)
    
    print(launch_config)
    
