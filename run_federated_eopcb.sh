
echo "【步骤1/5】正在检查GPU信息..."
nvidia-smi

# 设置环境变量
echo "【步骤2/5】正在设置MPI环境变量..."
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# 准备保存结果的目录
echo "【步骤3/5】正在创建结果保存目录..."
saving_path=$(pwd)/results/eo-pcb/yolov7/fedoptm
mkdir -p $saving_path
echo "结果目录已创建：$saving_path"

# 检查并下载预训练权重
echo "【步骤4/5】正在检查预训练权重..."
if [ ! -f weights/yolov7/yolov7_training.pt ]; then
    echo "权重文件不存在，正在下载..."
    wget -P weights/yolov7/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    # 下载后检查是否成功
    if [ -f weights/yolov7/yolov7_training.pt ]; then
        echo "权重下载成功！"
    else
        echo "权重下载失败！请手动下载后放到对应路径。"
        exit 1  # 下载失败则退出，避免后续报错
    fi
else
    echo "权重文件已存在，跳过下载。"
fi

echo "【步骤5/5】正在启动联邦学习（1服务器+5客户端）..."
mpirun -n 6 -x CUDA_VISIBLE_DEVICES=1 python federated/main.py \
    --nrounds 30 \
    --epochs 5 \
    --server-opt fedavgm \
    --server-lr 1.0 \
    --beta 0.1 \
    --architecture yolov7 \
    --weights weights/yolov7/yolov7_training.pt \
    --data data/eo-pcb6.yaml \
    --bsz-train 4 \
    --bsz-val 4 \
    --img 640 \
    --conf 0.001 \
    --iou 0.65 \
    --cfg yolov7/cfg/training/yolov7.yaml \
    --hyp data/hyps/hyp.scratch.clientopt.eo-pcb.yaml \
    --workers 4 \
    --use-immune-detection \
    --detection-threshold 0.5 \
    --trusted-clients 1,4,5\
    --client-attack-types 0,1.1,0,0
echo "联邦学习任务启动完成！"
# 备份实验结果
echo "【步骤6/6】正在备份实验结果..."
if [ -d "./experiments29" ]; then
    cp -r ./experiments29 $saving_path
    echo "结果备份成功：$saving_path/experiments29"
else
    echo "未找到experiments29目录，跳过备份。"
fi