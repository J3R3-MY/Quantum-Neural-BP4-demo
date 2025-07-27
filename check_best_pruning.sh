#!/bin/bash

src_base="../training_results/"  # Base path for source directories
dst_base="./training_results/"
dst_name="GB_46_2_800"                  # The name for all dest dirs
pattern="GB_46_2_800_*"                 # Change this to your actual pattern

for src_dir in "$src_base"/$pattern; do
    # Only directories
    [ -d "$src_dir" ] || continue

    echo "Processing $src_dir..."
    cp -r "$src_dir" "$dst_base/$dst_name"

    ./NBP_jupyter

    # Remove the copied directory
    rm -rf "$dst_base/$dst_name"
done
