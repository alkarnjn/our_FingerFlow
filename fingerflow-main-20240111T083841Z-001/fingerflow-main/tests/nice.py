import cv2
import numpy as np
from fingerflow.extractor import Extractor

extractor = Extractor("coarse_net", "fine_net", "classify_net", "core_net")

image = cv2.imread("/home/user/Downloads/fingerflow-main/scripts/extractor/102_4.tif")

extracted_minutiae = extractor.extract_minutiae(image)