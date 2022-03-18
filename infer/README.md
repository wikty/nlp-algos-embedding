## 一、模型导出

示例：把模型 *bert-base-chinese* 导出到 *export* 目录中，执行命令：`python modeling.py --model-name bert-base-chinese --onnx-path export`

模型导出过程中会有 ONNX 图校验+计算精度校验等信息打印出来，导出完成后有：（1）ONNX 文件 `model.onnx`，（2）模型初始化参数和tokenizer等信息配置文件 `config.json`

## 二、推理测试

脚本 `infer.py` 封装了在线推理计算逻辑（含预处理、模型预测以及后处理等），载入刚刚导出的模型进行测试：`python infer.py export/config.json`
