python ./main.py --backbone resnet50 --warmup 100 --bs 32 --epochs 50 --multistep 25 35  \
    --evaluation 0 1 5 10 15 20 25 30 35 40 45 50  \
    --amp --data /coco --save /coco/experiments/2labels_300_mixed/models --json-summary /coco/experiments/2labels_300_mixed/summary.json