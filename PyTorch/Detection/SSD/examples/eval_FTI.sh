echo 'folder' $1
WORK_DIR=$(readlink -f $1/../)
echo 'work dir' $WORK_DIR
for item in $(ls $WORK_DIR/models/*)
do
  echo $item
  python main.py  --backbone resnet50 --amp --ebs 8 --data /coco --mode evaluation --checkpoint $item >> $WORK_DIR/eval.txt
done
echo 'done!'
