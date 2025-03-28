CUDA_VISIBLE_DEVICES=1 python -u runIterativeTrainIORValEm.py --freeze_attention --savemodel --loadpath="data/model/ablation/ablation-valem.pt" --checkpointpath="data/model/fine-tune-look-twice/ablation-valem.pt" > fine-tune-ablation-valem.txt

CUDA_VISIBLE_DEVICES=2 python -u runIterativeTrainIResNet.py --freeze_attention --savemodel --loadpath="data/model/ablation/resnet.pt" --checkpointpath="data/model/fine-tune-look-twice/resnet.pt" > fine-tune-resnet.txt

CUDA_VISIBLE_DEVICES=2 python -u runIterativeTrainIOR.py --freeze_attention --savemodel --loadpath="data/model/ablation/ablation-augmentation.pt" --checkpointpath="data/model/fine-tune-look-twice/ablation-augmentation.pt" > fine-tune-ablation-augmentation.txt

CUDA_VISIBLE_DEVICES=3 python -u runIterativeTrainIOR.py --freeze_attention --savemodel --checkpointpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" > fine-tune-iormasked-nextclick.txt