import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def run_training():
    # Initialize the data loader and dataset split configuration layers
    datamodule = H3DataModule(nc_path="global_h3_res2_air_all_times.nc", batch_size=4)
    
    # Initialize your custom Neural Network graph model
    model = H3WeatherGCN(history_steps=2, learning_rate=1e-3)
    
    # Automatically save optimal weights files when validation loss improves
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-h3-weather-model",
        save_top_k=1,
        mode="min"
    )
    
    # Configure the PyTorch Lightning engine execution parameters
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto", # Automatically assigns GPU accelerators if available on your node
        devices="auto",     # Handles multi-GPU distribution automatically
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )
    
    print("Triggering Model Training Sequence...")
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    run_training()

