import torch
from unet.model import UNet3D
from trainer import Trainer
from dataset import BratsDataset
from torch.utils.data import DataLoader
from metrics import BCEDiceLoss


def main():
    train_ds = BratsDataset("data/train")
    valid_ds = BratsDataset("data/valid")
    batch_size = 2
    train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)
    valid_dl = DataLoader(valid_ds, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)

    device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(4, 1).to(device).float()

    trainer = Trainer(net=model,
                      train_dl=train_dl,
                      val_dl=valid_dl,
                      criterion=BCEDiceLoss(),
                      lr=5e-4,
                      accumulation_steps=batch_size,
                      batch_size=batch_size,
                      num_epochs=10,
                     )

    trainer.run()

if __name__ == "__main__":
    main()
