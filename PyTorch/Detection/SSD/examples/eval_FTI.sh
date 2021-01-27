echo $1
for item in $(ls $1/*)
do
  echo $item
  python main.py  --backbone resnet50 --amp --ebs 32 --data /coco --mode evaluation --checkpoint $item
done
echo 'done!'

