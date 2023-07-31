
'''
-- model_name
---- covmmd
---- fid128
---- kid128
'''


python fid_score_shapenet.py  ./results  ./model_name --dataset oo3d --reso 128


python kid_score.py --true ./results --fake ./model_name --dataset oo3d --reso 128


python compute_cd.py --gt_path ./results --gen_path ./model_name --n_points 2048
