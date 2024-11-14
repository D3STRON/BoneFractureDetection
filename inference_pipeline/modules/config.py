# set all model paths and model configuration values here

# model paths
yolo_path = "modules/models/yolo/best.pt"
resnet_path = "modules/models/resnet/model.pth"
rcnn_path = "place/holder/for/rcnn"
vit_path = "place/holder/for/vit"

# test path
# Path to the COCO annotation file
annotation_file = "validation_set/localization/coco_annotation/COCO_fracture_masks.json"
images_path = "validation_set/localization/COCO_images"

classification_test_data = "validation_set/classification/Images"

# configure task
task = "localization"