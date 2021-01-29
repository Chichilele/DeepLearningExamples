python ./main.py --backbone resnet50 --warmup 300 --bs 8 --epochs 50 --multistep 25 35  \
    --evaluation 0 1 2\
    --amp --data /coco --json-summary ./summary.json