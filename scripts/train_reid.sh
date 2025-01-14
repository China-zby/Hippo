cd ./pylib/Reid
for dataset_name in hippo; # jackson taipei square adventure flat;
do 
    CUDA_VISIBLE_DEVICES=0,1 python3 tools/train_net.py --config-file ./configs/Videodb/${dataset_name}.yml --num-gpus 2
done 