source /usr/local/miniconda3/bin/activate
# 设置acl环境变量等
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/usr/local/Ascend/thirdpart/aarch64/acllite:$PYTHONPATH

#
python3 main.py