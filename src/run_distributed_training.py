import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

def run_distributed_training():
    # 1. Initialize data loaders (Num_workers should match CPU cores per GPU)
    datamodule = H3DataModule(nc_path="global_h3_res2_air_all_times.nc", batch_size=4, num_workers=8)
    model = DistributedH3WeatherGCN(history_steps=2, learning_rate=1e-3)
    
    # 2. Configure DDP Strategy Settings for safe inter-node synchronization
    ddp_strategy = DDPStrategy(
        process_group_backend="nccl", # Use NVIDIA's high-speed NCCL backend
        find_unused_parameters=False, # Drastically reduces DDP gradient checking overhead
        static_graph=True             # Speeds up processing since our H3 mesh structure never changes
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="distributed-h3-weather-model",
        save_top_k=1,
        mode="min"
    )
    
    # 3. Setup Distributed Trainer Arguments
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices="auto",           # Uses all available GPUs on the allocated node(s)
        num_nodes=2,              # Set this to match your total number of allocated physical cluster nodes
        strategy=ddp_strategy,     # Activates Distributed Data Parallel orchestration
        callbacks=[checkpoint_callback],
        precision="16-mixed",     # Enables automatic mixed precision (AMP) to save memory and boost compute speed
        log_every_n_steps=10
    )
    
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    run_distributed_training()

