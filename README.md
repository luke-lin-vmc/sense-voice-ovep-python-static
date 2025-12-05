# About sense-voice-ovep-python-static
This Python pipeline is to show how to run [SenseVioce](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) ASR on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [RKNN implementation](https://github.com/k2-fsa/sherpa-onnx/tree/16d62b6a08f617c2bd6d21d411911c6462607f08/scripts/sense-voice/rknn) of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project 

Audio samples ("```en.mp3```", "```ja.mp3```", "```ko.mp3```", "```yue.mp3```" and "```zh.mp3```") are downloaded from [Hugging Face FunAudioLLM SenseVoice Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main/example)


## Key features
* Off-line (non-streaming) mode
* Multilingual speech recognition for Chinese, Cantonese, English, Japanese, and Korean
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download and export models
Visit https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main, download the following files
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
(openvino_venv) C:\Github\sense-voice-ovep-python-static>dir
 Volume in drive C is InstallTo
 Volume Serial Number is 76DF-BB22

 Directory of C:\Github\sense-voice-ovep-python-static

11/24/2025  03:08 PM    <DIR>          .
11/24/2025  02:58 PM    <DIR>          ..
11/24/2025  03:02 PM            11,203 am.mvn
11/24/2025  03:02 PM           377,341 chn_jpn_yue_eng_ko_spectok.bpe.model
11/24/2025  03:02 PM             1,855 config.yaml
11/24/2025  03:02 PM               396 configuration.json
11/23/2025  08:02 PM    <DIR>          example
11/23/2025  08:02 PM             4,802 export-onnx.py
11/24/2025  03:08 PM       928,985,316 model-5-seconds.onnx
11/24/2025  03:03 PM       936,291,369 model.pt
11/23/2025  08:02 PM             6,887 README.md
11/23/2025  08:02 PM               173 requirements.txt
11/23/2025  08:02 PM             7,171 test_onnx.py
11/24/2025  03:07 PM           340,949 tokens.txt
11/23/2025  08:02 PM            19,498 torch_model.py
11/24/2025  03:07 PM    <DIR>          __pycache__
              12 File(s)  1,866,046,960 bytes
               4 Dir(s)  225,240,248,320 bytes free
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
Run on CPU. Output text with punctuation and inverse text normalization.
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
### Sample log (device is NPU)
Without punctuation and inverse text normalization
```
(ovep_venv) C:\GitHub\sense-voice-ovep-python-static>python test_onnx.py --device NPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3
{'device': 'NPU', 'model': 'model-5-seconds.onnx', 'tokens': 'tokens.txt', 'wave': 'example\\zh.mp3', 'language': 'auto', 'use_itn': 0}
Device: OpenVINO EP with device = NPU
features.shape (93, 560)
features.shape (83, 560)
<|zh|><|NEUTRAL|><|Speech|><|woitn|>开 放 时 间 早 上 九 点 至 下 午 五 点
```
With punctuation and inverse text normalization
```
(ovep_venv) C:\GitHub\sense-voice-ovep-python-static>python test_onnx.py --device NPU --model model-5-seconds.onnx --token tokens.txt --wav example\zh.mp3 --use-itn 1
{'device': 'NPU', 'model': 'model-5-seconds.onnx', 'tokens': 'tokens.txt', 'wave': 'example\\zh.mp3', 'language': 'auto', 'use_itn': 1}
Device: OpenVINO EP with device = NPU
features.shape (93, 560)
features.shape (83, 560)
<|zh|><|NEUTRAL|><|Speech|><|withitn|>开 放 时 间 早 上 9 点 至 下 午 5 点 。
```
[Full log](https://github.com/luke-lin-vmc/sense-voice-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference
## System for validation
The pipeline has been validated on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
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

## Known issues
If the following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
This would be caused by that both ```onnxruntime``` and ```onnxruntime-openvino``` are installed. Solution is to remove both of them then re-install ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
Or simply to re-install ```onnxruntime-openvino``` if you would like to keep ```onnxruntime```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
