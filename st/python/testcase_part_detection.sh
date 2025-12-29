script_path_temp="$( cd "$(dirname $BASH_SOURCE)" ; pwd -P)"
project_path=${script_path_temp}/../../best_practices/contrib

function setEnv() {
  source /home/HwHiAiUser/Ascend/ascend-toolkit/set_env.sh
  export CANN_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
  export PATH=/usr/local/python3.7.5/bin:$PATH
}

function main() {
    setEnv
    #下载推理工具
    pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend'
    pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench'

    #下载数据
    cd ${project_path}/part_detection/inference/datasets
    if [ ! -f "${project_path}/part_detection/inference/datasets/mycoco-inference.zip" ];then
        wget -O ${project_path}/part_detection/inference/datasets/mycoco-inference.zip https://obs-9be7.obs.myhuaweicloud.com/models/Detection_Model_for_PyTorch/mycoco-inference.zip --no-check-certificate
    fi
    unzip -n -d ${project_path}/part_detection/inference/datasets/ ${project_path}/part_detection/datasets/mycoco-inference.zip

    #下载转换模型
    cd ${project_path}/part_detection/inference/
    if [ ! -f "${project_path}/part_detection/inference/yolov5l_mycoco1_bs4.om" ];then
        wget -O ${project_path}/part_detection/inference/yolov5l_mycoco1.onnx https://obs-9be7.obs.myhuaweicloud.com/models/Detection_Model_for_PyTorch/yolov5l_mycoco1.onnx --no-check-certificate
        bash onnx2om.sh --tag 6.0 --model yolov5l_mycoco1 --nms_mode nms_script --bs 4 --soc Ascend310
    fi
    python3 om_val.py --tag 6.0 \
                        --model=yolov5l_mycoco1_bs4.om \
                        --nms_mode nms_script \
                        --batch_size=4 \
                        --dataset Customised_dataset \
                        --visualization True \
                        --single_data True \
                        --data_path ./datasets/singleimg/ \
                        --cfg_file ./model4customised_dataset.yaml
    if [ $? -ne 0 ];then
        return 1
    fi
}
main