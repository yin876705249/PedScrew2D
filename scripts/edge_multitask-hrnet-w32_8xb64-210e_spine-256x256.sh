#!/bin/bash

CONFIG_PREFIX="configs/spine_2d_keypoint/topdown_heatmap/spine/edge_multitask-hrnet-w32_8xb64-210e_spine-256x256_fold_"
WORK_DIR_PREFIX="work_dirs/edge_multitask-hrnet-w32_8xb64-210e_spine-256x256_fold_"
SHOW_DIR="viz/"
RESULT_FILES=()
TEMP_FILE="edge_multitask-hrnet-w32_8xb64-210e_spine-256x256.txt"

# Clear the temporary file
> $TEMP_FILE

# Function to train and test a single fold
run_fold() {
    local fold=$1
    local gpu=$2

    echo "Processing fold $fold on GPU $gpu..."

    CONFIG_FILE="${CONFIG_PREFIX}${fold}.py"
    WORK_DIR="${WORK_DIR_PREFIX}${fold}/"

    # Train the model on the specified GPU
    CUDA_VISIBLE_DEVICES=$gpu python tools/train.py $CONFIG_FILE

    # Find the best .pth file automatically
    BEST_PTH_FILE=$(ls $WORK_DIR | grep 'best_OKS_epoch_' | sort | tail -n 1)

    if [ -z "$BEST_PTH_FILE" ]; then
        echo "No best model found for fold $fold. Skipping test..."
        return
    fi

    # Test the model on the specified GPU
    CUDA_VISIBLE_DEVICES=$gpu python tools/test.py $CONFIG_FILE "${WORK_DIR}${BEST_PTH_FILE}" --show-dir $SHOW_DIR

    # Find the latest test result directory
    LATEST_TEST_DIR=$(ls -td $WORK_DIR*/ | head -n 1)
    echo "Latest test directory for fold $fold: $LATEST_TEST_DIR"

    # Find the JSON file in the latest directory
    RESULT_JSON=$(ls -t $LATEST_TEST_DIR*.json | head -n 1)
    echo "Result JSON file for fold $fold: $RESULT_JSON"

    if [ -z "$RESULT_JSON" ]; then
        echo "No result JSON file found for fold $fold. Skipping..."
        return
    fi

    # Append the result JSON file path to the temporary file
    echo "$RESULT_JSON" >> $TEMP_FILE
}

# Run folds in pairs, using two GPUs
for i in {1..5..2}
do
    run_fold $i 0 &
    if [ $((i + 1)) -le 5 ]; then
        run_fold $((i + 1)) 1 &
    fi

    # Wait for both processes to complete before starting the next pair
    wait
done

echo "5-fold cross-validation completed."

# Read the result files from the temporary file
while IFS= read -r line
do
    RESULT_FILES+=("$line")
done < $TEMP_FILE

echo "Result Files: ${RESULT_FILES[@]}"

# Run Python script to calculate average metrics
python tools/crossval_average_metrics.py "${RESULT_FILES[@]}"

# Clean up the temporary file
rm -f $TEMP_FILE
