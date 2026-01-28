from oasis.train import train


args = ["--name", "angio", "--dataset_mode", "angio", "--gpu_ids", "0,2", "--dataroot", "angio_dataset", "--batch_size", "32"]
train(args)