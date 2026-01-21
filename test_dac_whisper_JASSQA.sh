#!/bin/bash

# Specify the path to the model checkpoint (.pt) file to be tested
CHECKPOINT_PATH=./ckpt/JASSQA_large_NISQA/best_model_epoch3_step30000.pt
# CHECKPOINT_PATH=./ckpt/JASSQA_large_VMC2023/model_epoch3_step24128.pt
# CHECKPOINT_PATH=./ckpt/JASSQA_medium_NISQA/model_epoch5_step41041.pt
# CHECKPOINT_PATH=./ckpt/JASSQA_medium_VMC2023/best_model_epoch3_step19968.pt

# Specify the type of feature extractor to be used when training this model.
AC_TOKENIZER_TYPE="dac"
SE_EXTRACTOR_TYPE="whisper_largev3" # optional "whisper_medium" or "whisper_largev3"

# Specify the path or name of the test dataset.
#    - If it's a local CSV file, use its full path, e.g., "/path/to/your/test.csv"

# NISQA
TEST_DATASET=("/path/to/database/NISQA_Corpus/NISQA_TEST_P501/NISQA_TEST_P501_file.csv")

# TEST_DATASET=("/path/to/database/NISQA_Corpus/NISQA_TEST_LIVETALK/NISQA_TEST_LIVETALK_file.csv")

# TEST_DATASET=("/path/to/database/NISQA_Corpus/NISQA_TEST_FOR/NISQA_TEST_FOR_file.csv")

# VMC23
# TEST_DATASET=("/path/to/database/VoiceMOS_2023_track3/VMC2023_TEST.csv")

# Tencent corpus
# TEST_DATASET=("/path/to/database/TencentCorups/withoutReverberationTrainDevMOS.csv")

# BVCC
# TEST_DATASET=("/path/to/database/BVCC/main/DATA/sets/test_mos_list.csv")



# Specify the root directory for the audio files.
AUDIO_ROOT_DIR="/path/to/database/NISQA_Corpus"
# AUDIO_ROOT_DIR="/path/to/database/TencentCorups"
# AUDIO_ROOT_DIR="/path/to/database/TCD_VOIP/tcdvoip"
# AUDIO_ROOT_DIR="/path/to/database/BVCC/main/DATA/wav"
# AUDIO_ROOT_DIR="/path/to/database/VoiceMOS_2023_track3"

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
