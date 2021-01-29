echo 'folder' $1
echo 'writing to file' $1/../eval.txt
for item in $(ls $1/*)
do
  echo $item
  python main.py  --backbone resnet50 --amp --ebs 32 --data /coco --mode evaluation --checkpoint $item >> $1/../eval.txt
done
echo 'done!'

