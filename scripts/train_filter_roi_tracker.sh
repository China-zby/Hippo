# amsterdam caldot1 caldot2 jackson shibuya uav warsaw
widths=(640 416 320 224 160)
heights=(352 256 192 128 96)
deviceID=1
device_numbers=2
data_flag=train
ddir=/home/lzp/otif-dataset/dataset
weightddir=/home/lzp/go-work/src/otifpipeline/weights
max_wait_epochs=100

for dataset in hippo;
do
    # for ((i=0,j=0; i<${#widths[@]} && j<${#heights[@]}; i++,j++))
    # do 
    #     device=$(((deviceID+1)%device_numbers))
    #     python ./pylib/Rois/CNN/train.py \
    #         --data_dir $ddir \
    #         --data_name $dataset --save_dir $weightddir \
    #         --width ${widths[i]} --height ${heights[j]} \
    #         --max_wait_epochs $max_wait_epochs
    #     deviceID=$((deviceID+1))

    #     device=$(((deviceID+1)%device_numbers))
    #     python ./pylib/Filters/CNN/train.py \
    #             --data_dir $ddir \
    #             --width ${widths[i]} --height ${heights[j]} \
    #             --data_name $dataset --save_dir $weightddir \
    #             --max_wait_epochs $max_wait_epochs --visual_label
    #     deviceID=$((deviceID+1))
    # done
    
    for skip_bound in 16 32 64 128 256; 
    do
        # tracker
        CUDA_VISIBLE_DEVICES=1 python ./pylib/Trackers/train_tracker.py --data_root $ddir \
                                                                        --save_root $weightddir \
                                                                        --data_name $dataset --max_wait_epochs $max_wait_epochs \
                                                                        --data_flag $data_flag \
                                                                        --cfg_path ./pylib/Trackers/tracker/cfg_${skip_bound}.json
        # # gnn
        # CUDA_VISIBLE_DEVICES=1 python ./pylib/Trackers/train_gnncnn.py --data_root $ddir/dataset \
        #                                                                --save_root $weightddir \
        #                                                                --data_name $dataset --max_wait_epochs $max_wait_epochs \
        #                                                                --data_flag $data_flag \
        #                                                                --cfg_path ./pylib/Trackers/gnncnn/cfg_${skip_bound}.json
    done
done