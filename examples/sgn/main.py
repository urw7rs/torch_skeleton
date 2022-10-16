import pytorch_lightning.cli as pl_cli

import litmodules
import datamodules

if __name__ == "__main__":
    cli = pl_cli.LightningCLI(save_config_overwrite=True)
