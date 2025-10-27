#!/bin/bash

# 定义基础路径和文件列表
BASE_OPT="options/RICE/"
LOG_PREFIX="log_"

# 要执行的yml文件列表
YML_FILES=(
    "rice1_dehazenet_model.yml"
    "rice1_nafnet_model.yml"
    "rice1_nafssr_model.yml"
    "rice2_dehazenet_model.yml"
    "rice2_nafnet_model.yml"
    "rice2_nafssr_model.yml"
)

# 遍历文件列表，依次执行
for yml in "${YML_FILES[@]}"; do
    # 提取日志文件名（去掉.yml后缀）
    log_file="${LOG_PREFIX}${yml%.yml}"

    echo "=== 开始执行: $yml ==="
    echo "日志将保存到: $log_file"

    # 执行命令（不放入后台，脚本会等待命令完成）
    nohup python train.py -opt "${BASE_OPT}${yml}" > "$log_file" 2>&1

    # 检查是否是最后一个文件，不是则等待10分钟
    if [ "$yml" != "${YML_FILES[-1]}" ]; then
        echo "$yml 执行完毕，等待10分钟后执行下一个..."
        sleep 600  # 10分钟 = 600秒
    fi
done

echo "所有命令已执行完毕"