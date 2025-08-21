from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2



model = GroundedSAM(ontology=CaptionOntology(
    {"cigarette butt": "megot", 
    "cigarette": "megot", 
    "cigarette filter": "megot", 
    "cigarette ash": "megot",
    "cigar butt": "megot",
    }))

result = model.predict("./source_images/test_close.jpg")
# I
plot(
    image=cv2.imread("./source_images/test_close.jpg"),
    classes=model.ontology.classes(),
    detections=result
)