CUDA_VISIBLE_DEVICES=0 python runTestAblationValEm.py --modelpath="data/model/fine-tune-look-twice/no_value_embedding.pt" --onecondition --conditionname='condition2' --ecc_mode --savename="modeldata-valem/fmodeldata_iterative_freeze-id1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestAblationValEm.py --modelpath="data/model/fine-tune-look-twice/no_value_embedding.pt" --onecondition --conditionname='condition1' --ecc_mode --savename="modeldata-valem/fmodeldata_iterative_freeze-ood1.npy" 

CUDA_VISIBLE_DEVICES=0 python runTestAblationValEm.py --modelpath="data/model/fine-tune-look-twice/no_value_embedding.pt" --onecondition --conditionname='condition3' --ecc_mode --savename="modeldata-valem/fmodeldata_iterative_freeze-id2.npy" 