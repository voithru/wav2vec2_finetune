# Wav2Vec2 Finetune and Inference
Wav2Vec2 finetune and inference code for IITP AI Grand Challenge

- 이 코드는 IITP 주관 인공지능그랜드챌린지 4차대회 트랙2에 사용한 코드를 공개한 것입니다.
- 이 코드는 huggingface에서 제공하는 Wav2Vec2 Finetuning에 대한 [sample code](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Fine_tuning_Wav2Vec2_for_English_ASR.ipynb) 를 참고하여 작성하였습니다.


## Finetune

### Dataset

```json lines
{"audio_path": "data/000.wav", "transcript": "제주도를 가려고 지금 김포공항에 와있습니다!"}
{"audio_path": "data/001.wav", "transcript": "오늘 일찍 와가지고 일찍 도착해서 1시간만 기다리면 돼요"}
{"audio_path": "data/002.wav", "transcript": "제주도 신라 호텔을 오면은 꼭 먹어봐야 되는 차돌박이 짬뽕이에요"}
```
- `config_train.yml`에서 `train_data_path`나 `test_data_path`로 입력받는 data json 파일은 위와 같은 형식으로 작성되어야 합니다. 
- `{"audio_path": "your_audio_path", "transcript": "audio_transcript"}`가 각 line에 입력되어야 합니다.

### Finetuning
```commandline
python main.py v1 train
```
- `config_train.yml`에서 `arguments`를 변경하실 수 있습니다.
- `pretrained_model_path`에 `huggingface`에 등록된 wav2vec2 pretrained model을 입력하여 사용할 수 있습니다.
- `pretrained_model_path`에 local checkpoint 경로를 입력하여 사용할 수도 있습니다.
- 다른 parameter들은 [transformer documents](https://huggingface.co/docs/transformers/main_classes/trainer) 를 참조하시길 바랍니다.


## Inference

###Inference for one checkpoint
```commandline
python main.py v1 predict one
```

###Inference for many checkpoint
```commandline
python main.py v1 predict many
```

- `config_predict.yml`에서 `arguments`를 변경하실 수 있습니다.