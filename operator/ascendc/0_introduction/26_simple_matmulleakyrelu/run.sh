#!/bin/bash
rm -rf build; mkdir -p build; cd build
cmake ..; make -j
python3 ../scripts/gen_data.py
./demo
python3 ../scripts/verify_result.py output/output.bin output/golden.bin
