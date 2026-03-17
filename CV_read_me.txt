download https://drive.google.com/file/d/1TFRvIWbsjxZm9DTx1zr6F9N0M-XE3JMo/view?usp=share_link the model
download infer_stream.py
download qevd_pose.yml
    make env with conda using yml file or with your env manager

set up folders as 

checkpoints/qevd_full_m_v2/epoch_1250.pt
run infer infer_stream

python infer_stream.py --ckpt checkpoints/qevd_full_m_v2/epoch_1250.pt --source 0 --device cpu --show --window 128 --stride 64
