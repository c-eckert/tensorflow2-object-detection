```
python train.py --img 640 --cfg models/yolov5s.yaml --hyp data/hyps/hyp.scratch-high.yaml --batch -1 --epochs 100 --data data/rdd5.yaml --weights yolov5s.pt --workers 32 --name s_5highaug
python train.py --img 640 --cfg models/yolov5s.yaml --hyp data/hyps/hyp.scratch-high.yaml --batch -1 --epochs 100 --data data/rddCzJpNoUs.yaml --weights yolov5s.pt --workers 32 --name s_CJNU

python detect.py --weights runs/train/s_CJNU/weights/best.pt --img 640 --conf 0.20 --name s_CJNU --source ~/Bilder/VID_20220821_094242.mp4

python export.py --weights runs/train/s_highaug/weights/best.pt --include tflite --int8

python train.py --img 640 --hyp data/hyps/hyp.scratch-high.yaml --batch -1 --epochs 100 --data data/rdd_Japan.yaml --weights yolov5s.pt --workers 32 --name s_Japan
```
