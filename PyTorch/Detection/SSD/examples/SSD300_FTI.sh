python ./main.py --backbone resnet50 --warmup 10 --bs 64 --epochs 50 --multistep 25 35  \
    --evaluation 0 1 5 10 15 20 25 30 35 40 45 50 \
    --amp --data /coco --save /coco/experiments/2labels_512/models --json-summary /coco/experiments/2labels_512/summary.json