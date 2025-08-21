from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
#from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations

base_model = GroundedSAM(ontology=CaptionOntology(
    {"cigarette butt": "megot", 
    "cigarette": "megot", 
    "cigarette filter": "megot", 
    "cigarette ash": "megot",
    "cigar butt": "megot",
    }))



base_model.label("./source_images", extension=".jpg")

#target_model = YOLOv8("yolov8n.pt")
#target_model.train("./labeled-images/data.yaml", epochs=200)
