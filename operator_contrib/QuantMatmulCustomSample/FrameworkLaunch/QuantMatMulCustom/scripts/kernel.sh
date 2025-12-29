#!/bin/bash

OUT_DEBUG_DIR=$1
FINAL_SO_PATH="$OUT_DEBUG_DIR/../libascendc_custom_kernels.so"
chmod 775 $OUT_DEBUG_DIR

out_binary_files=$(find $OUT_DEBUG_DIR -type f -name "*.o")
for out_binary_file in $out_binary_files; do
    if [ -f "${out_binary_file%.*}.json" ]; then
        continue
    fi
    echo "process $out_binary_file to libascendc_custom_kernels.so"
    if [ -f $FINAL_SO_PATH ]; then 
        ld.lld  -m aicorelinux -r -Ttext=0 -x $FINAL_SO_PATH $out_binary_file -static -o $FINAL_SO_PATH
    else
        ld.lld  -m aicorelinux -r -Ttext=0 -x $out_binary_file -static -o $FINAL_SO_PATH
    fi
done