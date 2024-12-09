# set all model paths and model configuration values here

# model paths
yolo_path = "modules/models/yolo/best.pt"
resnet_path = "modules/models/resnet/model.pth"
rcnn_path = "place/holder/for/rcnn"
vit_path = "place/holder/for/vit"
detr_path = './modules/models/detr/epoch=22-step=14835.ckpt'

# test path
# Path to the COCO annotation file
annotation_file = "validation_set/localization/coco_annotation/_annotations.coco.json"
images_path = "validation_set/localization/COCO_images"

classification_test_data = "validation_set/classification/valid_compiled.csv"

# configure task
task = "localization"