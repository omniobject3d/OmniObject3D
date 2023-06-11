#!/bin/bash

folder_path=$1

cd "$folder_path"

for subfile in *; do
    if [[ $subfile == *.tar.gz ]]; then
        subfolder=${subfile%.tar.gz}
        mkdir $subfolder
        tar -xf $subfile -C $subfolder
    fi

    echo "$subfile extracted to $subfolder"
done
