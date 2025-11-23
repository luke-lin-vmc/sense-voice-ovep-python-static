# About sense-voice-ovep-python-static
This Python pipeline is to show how to run SenseVioce ASR on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [RKNN implementation](https://github.com/k2-fsa/sherpa-onnx/tree/16d62b6a08f617c2bd6d21d411911c6462607f08/scripts/sense-voice/rknn) of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project 

Audio samples ("```en.mp3```, ```ja.mp3```, ```ko.mp3```, ```yue.mp3``` and ```zh.mp3```") are downloaded from [Hugging Face FunAudioLLM SenseVoice Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main/example)


## Key features
* Off-line (non-streaming) mode
* Support Chinese (zh) only
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download and export models
Visit https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main, download the following 
files
```
am.mvn
chn_jpn_yue_eng_ko_spectok.bpe.model
config.yaml
configuration.json
model.pt
```
Run the following commands to export models
```
pip install -r requirements.txt
python export-onnx.py --input-len-in-seconds 5
```
* As NPU does not support dynamic input shape, it is required to convert to static by specifying a fixed input length. You may configure the input length by setting  "```--input-len-in-seconds <length>```" per your requirement.

The following models (```*.onnx```) will be exported under the same directory
```
model-5-seconds.onnx
```
The project directory should look like
```
(openvino_venv) C:\Github\paraformer-zh-ovep-python-static>dir
 Volume in drive C is InstallTo
 Volume Serial Number is 76DF-BB22

 Directory of C:\Github\paraformer-zh-ovep-python-static

11/21/2025  11:51 AM    <DIR>          .
11/20/2025  04:20 PM    <DIR>          ..
11/21/2025  11:32 AM           179,712 0.wav
11/21/2025  11:32 AM           165,042 1.wav
11/21/2025  11:32 AM           144,922 2.wav
11/21/2025  11:44 AM            11,203 am.mvn
11/21/2025  11:44 AM             2,509 config.yaml
11/21/2025  11:44 AM               472 configuration.json
11/21/2025  11:50 AM       228,460,151 decoder-5-seconds.onnx
11/21/2025  11:50 AM       632,885,122 encoder-5-seconds.onnx
11/21/2025  11:32 AM             1,169 export_decoder_onnx.py
11/21/2025  11:32 AM             5,189 export_encoder_onnx.py
11/21/2025  11:32 AM             1,510 export_predictor_onnx.py
11/21/2025  11:45 AM       880,502,012 model.pt
11/21/2025  11:51 AM         3,152,772 predictor-5-seconds.onnx
11/21/2025  11:32 AM             6,446 README.md
11/21/2025  11:32 AM               120 requirements.txt
11/21/2025  11:45 AM         8,287,834 seg_dict
11/21/2025  11:32 AM            10,969 test_onnx.py
11/21/2025  11:45 AM            93,676 tokens.json
11/21/2025  11:32 AM            44,688 torch_model.py
11/21/2025  11:50 AM    <DIR>          __pycache__
              19 File(s)  1,753,955,518 bytes
               3 Dir(s)  225,971,204,096 bytes free
```
## Run
Usage
```
usage: test_onnx.py [-h] [--device DEVICE] --model MODEL --tokens TOKENS --wave WAVE [--language LANGUAGE]
                    [--use-itn USE_ITN]

options:
  -h, --help           show this help message and exit
  --device DEVICE      Execution device. Use 'CPU', 'GPU', 'NPU' for OpenVINO. If not specified, the default
                       CPUExecutionProvider will be used. (default: None)
  --model MODEL        Path to model.onnx (default: None)
  --tokens TOKENS      Path to tokens.txt (default: None)
  --wave WAVE          The input wave to be recognized (default: None)
  --language LANGUAGE  the language of the input wav file. Supported values: zh, en, ja, ko, yue, auto (default:
                       auto)
  --use-itn USE_ITN    1 to use inverse text normalization. 0 to not use inverse text normalization (default: 0)
  ```

Run on CPU
```
python test_onnx.py --device CPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3
```
Run on CPU, output result includes punctuation and inverse text normalization.
```
python test_onnx.py --device CPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3 --use-itn 1
```
Run on GPU
```
python test_onnx.py --device GPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3
```
Run on NPU
```
python test_onnx.py --device NPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3
```
:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested devices
The pipeline has been verified working on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 140V GPU, driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4404 (11/7/2025)```
### Result
| Sample             | CPU | GPU | NPU |
|--------------------|-----|-----|-----|
| English (en.mp3)   | OK  | OK  | OK  |
| Japanese (ja.mp3)  | OK  | OK  | OK  |
| Korean (ko.mp3)    | OK  | OK  | OK  |
| Cantonese (yue.mp3)| OK  | OK  | OK  |
| Chinese (zh.mp3)   | OK  | OK  | OK  |

### Sample log (device is NPU)
```
(ovep_venv) C:\GitHub\sense-voice-ovep-python-static>python test_onnx.py --device NPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3
{'device': 'NPU', 'model': 'model-5-seconds.onnx', 'tokens': 'tokens.txt', 'wave': 'example\\zh.mp3', 'language': 'auto', 'use_itn': 0}
Device: OpenVINO EP with device = NPU
features.shape (93, 560)
features.shape (83, 560)
<|zh|><|NEUTRAL|><|Speech|><|woitn|>开 放 时 间 早 上 九 点 至 下 午 五 点
```
[Full log](https://github.com/luke-lin-vmc/sense-voice-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
The following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
Solution is to simply re-install ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
