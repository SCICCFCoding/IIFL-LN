echo "【Step 1/5】Checking GPU information..."
nvidia-smi

echo "【Step 2/5】Setting MPI environment variables..."
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

echo "【Step 3/5】Creating result saving directory..."
saving_path=$(pwd)/results/eo-pcb/yolov7/fedoptm
mkdir -p $saving_path
echo "Result directory created: $saving_path"


echo "【Step 4/5】Checking pre-trained weights..."
if [ ! -f weights/yolov7/yolov7_training.pt ]; then
    echo "Weight file does not exist, downloading..."
    wget -P weights/yolov7/ https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt

    if [ -f weights/yolov7/yolov7_training.pt ]; then
        echo "Weight download successful!"
    else
        echo "Weight download failed! Please download it manually and place it in the corresponding path."
        exit 1  
    fi
else
    echo "Weight file already exists, skipping download."
fi

echo "【Step 5/5】Starting federated learning (1 server + 5 clients)..."
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
echo "Federated learning task started successfully!"

echo "【Step 6/6】Backing up experiment results..."
if [ -d "./experiments29" ]; then
    cp -r ./experiments29 $saving_path
    echo "Result backup successful: $saving_path/experiments29"
else
    echo "experiments29 directory not found, skipping backup."
fi