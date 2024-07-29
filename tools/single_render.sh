#! /bin/bash
export DISPLAY=:0.0 && \
blender -b -P ./blender_script.py -- --obj_path /local_home/shenqiuhong/omni3d/raw_scan/antique/antique_004 \
--output /local_home/shenqiuhong/render_test \