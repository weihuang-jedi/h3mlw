import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class H3DataModule(pl.LightningDataModule):
    def __init__(self, nc_path="../data/global_h3_res2_air_all_times.nc", batch_size=8, num_workers=4):
        super().__init__()
        self.nc_path = nc_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load the unified dataset sequence
        full_dataset = GlobalH3WeatherDataset(self.nc_path)
        
        # Group time steps into clean, isolated sets (e.g., 80% train, 10% val, 10% test)
        total = len(full_dataset)
        train_sz = int(total * 0.8)
        val_sz = int(total * 0.1)
        test_sz = total - train_sz - val_sz
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_sz, val_sz, test_sz]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

