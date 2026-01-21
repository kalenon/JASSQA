#!/bin/bash

# Specify the path to the model checkpoint (.pt) file to be tested
# CHECKPOINT_PATH=/data/home/wangchaoyang/code/SQA/Tok-MetricNet_whistoken/results/dac_whisperlarge_Quality_NISQA_latentembed3fusionmeantime_bs1_1loss_128dim_step_scoreqmaskdualpathcross8_DNSptforNISQA_lowstep_dp04all/best_model_epoch3_step30000.pt
# CHECKPOINT_PATH=/data/home/wangchaoyang/code/SQA/Tok-MetricNet_whistoken/results/dac_whispermed_Quality_NISQA_latentembed3fusionmeantime_bs1_1loss_128dim_step_scoreqmaskdualpathcross8_DNSforNISQA_lowstep_dp04all_exdata_10/model_epoch5_step41041.pt
CHECKPOINT_PATH=/data1/wangchaoyang/code/JASSQA/ckpt/best_model_epoch3_step30000.pt

# Specify the type of feature extractor to be used when training this model.
AC_TOKENIZER_TYPE="dac"
SE_EXTRACTOR_TYPE="whisper_largev3"

# Specify the path or name of the test dataset.
#    - If it's a local CSV file, use its full path, e.g., "/path/to/your/test.csv"

# TEST_DATASET=("/data/home/wangchaoyang/database/NISQA_Corpus/NISQA_TEST_P501/NISQA_TEST_P501_file.csv")
# TEST_WHISPER_ROOT_DIR=("/data/home/wangchaoyang/code/SQA/MOSA-Net-Cross-Domain-main/MOSA_Net+/MOSA-Net_Plus_Torch/Test_SSL_NISQA_P501_Feat_Whisperv3")

# TEST_DATASET=("/data/home/wangchaoyang/database/NISQA_Corpus/NISQA_TEST_LIVETALK/NISQA_TEST_LIVETALK_file.csv")
# TEST_WHISPER_ROOT_DIR=("/data/home/wangchaoyang/code/SQA/MOSA-Net-Cross-Domain-main/MOSA_Net+/MOSA-Net_Plus_Torch/Test_SSL_NISQA_LIVETALK_Feat_Whisperv3")

TEST_DATASET=("/data/home/wangchaoyang/database/NISQA_Corpus/NISQA_TEST_FOR/NISQA_TEST_FOR_file.csv")
# TEST_WHISPER_ROOT_DIR=("/data/home/wangchaoyang/code/SQA/MOSA-Net-Cross-Domain-main/MOSA_Net+/MOSA-Net_Plus_Torch/Test_SSL_NISQA_FOR_Feat_Whisperv3")

# TEST_DATASET=("/data/home/wangchaoyang/database/NISQA_Corpus/NISQA_VAL_SIM/NISQA_VAL_SIM_file.csv")
# TEST_WHISPER_ROOT_DIR=("/data/home/wangchaoyang/code/SQA/MOSA-Net-Cross-Domain-main/MOSA_Net+/MOSA-Net_Plus_Torch/Val_SSL_NISQA_Feat_Whisperv3")

# Tencent corpus
# TEST_DATASET=("/data/home/wangchaoyang/database/TencentCorups/withoutReverberationTrainDevMOS.csv")
# TEST_DATASET=(/data/home/wangchaoyang/code/SQA/MOSA-Net-Cross-Domain-main/MOSA_Net+/MOSA-Net_Plus_Torch/TENCENT_TXT/Test_TENCENT.csv)

# TCP VoIP
# TEST_DATASET=("/data/home/wangchaoyang/database/TCD_VOIP/tcdvoip/tcdvoip_ood_test.csv")

# BVCC
# TEST_DATASET=("/data/home/wangchaoyang/database/BVCC/main/DATA/sets/test_mos_list.csv")

# Specify the root directory for the audio files.
AUDIO_ROOT_DIR="/data/home/wangchaoyang/database/NISQA_Corpus"
# AUDIO_ROOT_DIR="/data/home/wangchaoyang/database/TencentCorups"
# AUDIO_ROOT_DIR="/data/home/wangchaoyang/database/TCD_VOIP/tcdvoip"
# AUDIO_ROOT_DIR="/data/home/wangchaoyang/database/BVCC/main/DATA/wav"

TARGET_METRIC="quality"

BATCH_SIZE=1

OUTPUT_CSV="test_predictions.csv"

OUTPUT_PLOT='scatterplot.png'


echo "Commencing model testing..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Test set: $TEST_DATASET"

CMD="CUDA_VISIBLE_DEVICES=0 \
    python test_dac_whisper_JASSQA.py \
    --checkpoint_path \"$CHECKPOINT_PATH\" \
    --ac_tokenizer_type \"$AC_TOKENIZER_TYPE\" \
    --sematic_type \"$SE_EXTRACTOR_TYPE\" \
    --test_dataset \"${TEST_DATASET[@]}\" \
    --audio_root_dir \"$AUDIO_ROOT_DIR\" \
    --target_metric \"$TARGET_METRIC\" \
    --batch_size $BATCH_SIZE"

if [ -n "$OUTPUT_CSV" ]; then
    CMD="$CMD --output_csv \"$OUTPUT_CSV\""
fi

if [ -n "$OUTPUT_PLOT" ]; then
    CMD="$CMD --plot \"$OUTPUT_PLOT\""
fi

echo "Executing command: $CMD"
eval $CMD

echo "Test script execution completed."
