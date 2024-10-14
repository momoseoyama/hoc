#!/bin/bash

SAVE_DIR="./table2_reproduction"

for topk in {1..5}
do
    echo "Running evaluation for topk = $topk"
    python table2_gpt-evaluation.py run --embedding ICA --topk $topk --dims 100 --model gpt-4o-mini --save_dir $SAVE_DIR
done

echo "All evaluations completed. Use the display command to view results."