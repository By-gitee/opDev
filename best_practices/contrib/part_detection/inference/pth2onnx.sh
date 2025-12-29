##### 参数设置
GETOPT_ARGS=`getopt -o 'h' -al tag:,model:,nms_mode: -- "$@"`
eval set -- "$GETOPT_ARGS"
while [ -n "$1" ]
do
    case "$1" in
        --tag) tag=$2; shift 2;;
        --model) model=$2; shift 2;;
        --nms_mode) nms_mode=$2; shift 2;;
        --) break ;;
    esac
done

if [[ -z $tag ]]; then tag=6.1; fi
if [[ -z $model ]]; then model=yolov5s; fi
if [[ -z $nms_mode ]]; then nms_mode=nms_op; fi

args_info="=== pth2onnx args === \n tag: $tag \n model: $model \n nms_mode: $nms_mode"
echo -e $args_info

##### 环境准备
export PYTHONPATH="$PWD"
git checkout . && git checkout v${tag}  ## 切换到目标tag

##### 方式一 nms后处理脚本
if [[ $nms_mode == nms_script ]] ; then
    echo "nms后处理脚本"
    git apply common/patch/v${tag}.patch --exclude models/yolo.py ## 兼容性修改，支持导出不同版本的YOLOv5
    export=`find ./ -name export.py`  ## 查找export.py文件路径，不同版本路径不一样
    python3 ${export} --weights=${model}.pt --opset=11 --dynamic || exit 1  # 导出onnx模型
fi

##### 方式二 nms后处理算子
if [[ $nms_mode == nms_op ]] ; then
    echo "nms后处理算子"
    git apply common/patch/v${tag}.patch  ## 兼容性修改，支持导出不同版本的YOLOv5，Detect部分修改以适配添加nms后处理算子
    export=`find ./ -name export.py`  ## 查找export.py文件路径，不同版本路径不一样
    python3 ${export} --weights=${model}.pt --opset=11 --dynamic || exit 1  # 导出onnx模型
    python3 common/util/add_nms.py --pt=${model}.pt --onnx=${model}.onnx --cfg=model.yaml || exit 1  ## 修改模型，添加nms后处理算子
fi

echo -e "pth导出onnx模型 Success \n"
