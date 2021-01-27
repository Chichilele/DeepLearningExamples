python ./main.py --backbone resnet50 --warmup 10 --bs 64 --epochs 50 --multistep 15 25  \
    --evaluation 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 35 40 45 50 \
    --amp --data /coco --save /coco/experiments/2labels_mixed_dataset/models --json-summary /coco/experiments/2labels_mixed_dataset/summary.json