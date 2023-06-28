folders="./data/clip_embds/uhd/for_webdataset/face_images/*"
for folder in $folders; do
    tar -cvf $folder.tar $folder
done