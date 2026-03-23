import torch
import os
import omegaconf
from src.datasets.nad_dataset import NADDataModule, NADDatasetInfos
from src.diffusion_model_discrete_reg_free import DiscreteDenoisingDiffusion
from src.metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from src.diffusion.extra_features import DummyExtraFeatures

import wandb

def run_dry_run():
    print("====== Starting Dry-Run Verification ======")
    wandb.init(mode="disabled")
    try:
        cfg = omegaconf.OmegaConf.create({
            "dataset": {
                "name": "nad",
                "datadir": "NAD_triplet_dataset.jsonl",
                "remove_h": False,
                "embeddings_file": None
            },
            "train": {
                "batch_size": 2,
                "num_workers": 0,
                "lr": 0.0001,
                "weight_decay": 0.0
            },
            "model": {
                "type": "discrete",
                "diffusion_steps": 50,
                "diffusion_noise_schedule": "cosine",
                "n_layers": 2,
                "hidden_mlp_dims": {"X": 64, "E": 64, "y": 64},
                "hidden_dims": {"dx": 64, "de": 32, "dy": 32, "n_head": 4, "dim_ffX": 64, "dim_ffE": 32},
                "lambda_train": [1.0, 1.0],
                "transition": "uniform"
            },
            "general": {
                "name": "test_dryrun",
                "log_every_steps": 10,
                "number_chain_steps": 10
            }
        })
        
        print("1. Initializing DataModule...")
        datamodule = NADDataModule(cfg)
        dataset_infos = NADDatasetInfos(datamodule, cfg)
        
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
        
        print("2. Computing input output dims...")
        dataset_infos.compute_input_output_dims(
            datamodule=datamodule, 
            extra_features=extra_features, 
            domain_features=domain_features
        )
        
        print(f"Input dimensions calculated: {dataset_infos.input_dims}")
        print(f"Output dimensions calculated: {dataset_infos.output_dims}")
        
        train_metrics = TrainAbstractMetricsDiscrete()
        sampling_metrics = None
        visualization_tools = None

        model_kwargs = {
            'dataset_infos': dataset_infos, 
            'train_metrics': train_metrics,
            'sampling_metrics': sampling_metrics, 
            'visualization_tools': visualization_tools,
            'extra_features': extra_features, 
            'domain_features': domain_features
        }

        print("3. Initializing Diffusion Model...")
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
        model = model.to('cpu')
        
        print("4. Loading a batch from DataLoader...")
        batch = next(iter(datamodule.train_dataloader()))
        batch = batch.to('cpu')
        
        print(f"   [Batch] Child Graph X={batch.x.shape}, Edge Index={batch.edge_index.shape}, Edge Attr={batch.edge_attr.shape}")
        print(f"   [Batch] Parent Graph X_parent={batch.X_parent.shape}, E_parent={batch.E_parent.shape}")
        print(f"   [Batch] Text condition y={batch.y.shape}")
        
        optimizer = model.configure_optimizers()
        optimizer.zero_grad()
        
        print("5. Running Forward Pass...")
        loss_dict = model.training_step(batch, i=0)
        loss = loss_dict['loss']
        print(f"   -> Loss computed: {loss.item()}")
        
        print("6. Running Backward Pass...")
        loss.backward()
        print("   -> Backward Pass completed without errors!")
        
        print("====== Dry-Run Verification Successful! ======")

    except Exception as e:
        print(f"Dry-run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_dry_run()
