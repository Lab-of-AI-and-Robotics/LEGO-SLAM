#!/bin/bash

# Usage: bash run_scannet.sh /path/to/Scannet
if [ -z "$1" ]; then
    echo "Usage: bash run_scannet.sh <dataset_path>"
    echo "Example: bash run_scannet.sh /path/to/Scannet"
    exit 1
fi

OUTPUT_PATH="experiments"
DATASET_PATH="$1"

str_pad() {

  local pad_length="$1" pad_string="$2" pad_type="$3"
  local pad length llength offset rlength

  pad="$(eval "printf '%0.${#pad_string}s' '${pad_string}'{1..$pad_length}")"
  pad="${pad:0:$pad_length}"

  if [[ "$pad_type" == "left" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "${pad:0:$length}$line"
    done

  elif [[ "$pad_type" == "both" ]]; then

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      llength="$(( length / 2 ))"
      offset="$(( llength + ${#line} ))"
      rlength="$(( llength + (length % 2) ))"
      echo -n "${pad:0:$llength}$line${pad:$offset:$rlength}"
    done

  else

    while read line; do
      line="${line:0:$pad_length}"
      length="$(( pad_length - ${#line} ))"
      echo -n "$line${pad:${#line}:$length}"
    done

  fi
}

run_()
{
    local dataset=$1
    local config=$2
    local result_txt=$3
    local keyframe_th=$4
    local knn_maxd=$5
    local overlapped_th=$6
    local max_correspondence_distance=$7
    local trackable_opacity_th=$8
    local overlapped_th2=$9
    local downsample_rate=${10}
    local post_training_iter=${11}
    local eval_ratio=${12}
    local edge_weight=${13}
    local n_trackable_keyframes=${14}
    local pose_lr_rate=${15}
    local loopclosing_global_correspondence_distance=${16}
    local loopclosing_local_correspondence_distance=${17}
    local loop_constraint_noise=${18}
    
    echo "run $dataset"
    echo "run $dataset" >> ${result_txt}
    python -W ignore lego_slam.py --dataset_path $DATASET_PATH/$dataset\
                                    --config $config\
                                    --output_path $OUTPUT_PATH/$dataset/init/\
                                    --keyframe_th $keyframe_th\
                                    --keyframe_freq $keyframe_freq\
                                    --sparse_weight $sparse_weight\
                                    --target_cov $target_cov\
                                    --source_cov_norm $source_cov_norm\
                                    --knn_maxd $knn_maxd\
                                    --overlapped_th $overlapped_th\
                                    --max_correspondence_distance $max_correspondence_distance\
                                    --trackable_opacity_th $trackable_opacity_th\
                                    --overlapped_th2 $overlapped_th2\
                                    --downsample_rate $downsample_rate\
                                    --post_training_iter $post_training_iter\
                                    --eval_ratio $eval_ratio\
                                    --save_results \
                                    --enable_loop_closing \
                                    --system_fps_limit 15.0 \
                                    --speedup \
                                    --edge_weight ${edge_weight} \
                                    --n_trackable_keyframes $n_trackable_keyframes \
                                    --pose_lr_rate $pose_lr_rate \
                                    --max_mapping_keyframes 50 \
                                    --loop_constraint_noise $loop_constraint_noise \
                                    --loopclosing_global_correspondence_distance $loopclosing_global_correspondence_distance \
                                    --loopclosing_local_correspondence_distance $loopclosing_local_correspondence_distance \
                                    --semantic_feature_init \
                                    --pretrained_encoder_path "saved/cnn_encoder_best.pth" \
                                    --pretrained_decoder_path "saved/cnn_decoder_best.pth" >> ${result_txt}
    wait
}

run_scannet()
{
    local result_txt=$1
    local keyframe_th=$2
    local knn_maxd=$3
    local overlapped_th=$4
    local max_correspondence_distance=$5
    local trackable_opacity_th=$6
    local overlapped_th2=$7
    local downsample_rate=$8
    local post_training_iter=$9
    local eval_ratio=${10}
    local edge_weight=${11}
    local n_trackable_keyframes=${12}
    local pose_lr_rate=${13}
    local loopclosing_global_correspondence_distance=${14}
    local loopclosing_local_correspondence_distance=${15}
    local loop_constraint_noise=${16}
    
    run_ "scene0000_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
    run_ "scene0059_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
    run_ "scene0106_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
    run_ "scene0169_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
    run_ "scene0181_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
    run_ "scene0207_00" "configs/ScanNet/caminfo.txt" "$result_txt" "$keyframe_th" "$knn_maxd" "$overlapped_th" "$max_correspondence_distance" "$trackable_opacity_th" "$overlapped_th2" "$downsample_rate" "$post_training_iter" "$eval_ratio" "$edge_weight" "$n_trackable_keyframes" "$pose_lr_rate" "$loopclosing_global_correspondence_distance" "$loopclosing_local_correspondence_distance" "$loop_constraint_noise"
}

txt_file="scannet_results.txt"

overlapped_th=0.001
max_correspondence_distance=0.05
knn_maxd=99999.0
trackable_opacity_th=0.09
overlapped_th2=1e-4
downsample_rate=5
keyframe_th=0.85
post_training_iter=0
eval_ratio=1.0
edge_weight=10.0
n_trackable_keyframes=25
pose_lr_rate=0.5
loopclosing_global_correspondence_distance=0.5
loopclosing_local_correspondence_distance=0.05
loop_constraint_noise=1e-3
keyframe_freq=30
sparse_weight=1.0
target_cov=additional
source_cov_norm=max

run_scannet $txt_file $keyframe_th $knn_maxd $overlapped_th $max_correspondence_distance \
  $trackable_opacity_th $overlapped_th2 $downsample_rate $post_training_iter $eval_ratio $edge_weight $n_trackable_keyframes $pose_lr_rate $loopclosing_global_correspondence_distance $loopclosing_local_correspondence_distance $loop_constraint_noise
