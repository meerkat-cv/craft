import cv2

from craft_detector import CraftDetector


controller = CraftDetector("../../data/craft_text_detector/craft_mlt_25k.pth")
Im = cv2.imread("../../data/images/image_0608.jpg")
controller.test_net(Im)