CUDA_VISIBLE_DEVICES=0 python runTestAblationResNet.py --modelpath="data/model/fine-tune-look-twice/resnet.pt" --onecondition --conditionname='condition2' --ecc_mode --savename="modeldata-resnet/fmodeldata_iterative_freeze-id1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestAblationResNet.py --modelpath="data/model/fine-tune-look-twice/resnet.pt" --onecondition --conditionname='condition1' --ecc_mode --savename="modeldata-resnet/fmodeldata_iterative_freeze-ood1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestAblationResNet.py --modelpath="data/model/fine-tune-look-twice/resnet.pt" --onecondition --conditionname='condition3' --ecc_mode --savename="modeldata-resnet/fmodeldata_iterative_freeze-id2.npy" 