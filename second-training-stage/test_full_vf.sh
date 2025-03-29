CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='condition2' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-id1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='condition1' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='condition3' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-id2.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood2' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood2.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood3' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood3.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood4' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood4.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood5' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood5.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood6' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood6.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood7' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood7.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestIterativeTrain.py --freeze --modelpath="data/model/fine-tune-look-twice/iormasked-nextclick.pt" --iterative --onecondition --conditionname='ood8' --ecc_mode --savename="modeldata-nextclick/fmodeldata_iterative_freeze-ood8.npy" 