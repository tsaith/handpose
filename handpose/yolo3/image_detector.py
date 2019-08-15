from handpose.utils.file import get_file_list
from handpose.utils.image import draw_detection, get_rgb_colors

from torchvision import transforms
from PIL import Image
import os

class ImageDetector:

    def __init__(self, model, classes):

        self.model = model
        self.classes = classes
        self.colors = get_rgb_colors()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def detect(self, image):

        # Convert into tensor
        image_pt = self.transform(image)
        image_pt = image_pt.unsqueeze(0)

        detections = self.model.detect(image_pt)

        return detections

    def draw_image(self, image):

        # Make detections
        detections = self.detect(image)

        # Draw detections
        if detections is not None:
            width, height = image.size
            draw_detection(image, width, height, detections, self.classes, self.colors)

        return image

    def export_images(self, image_dir='./images', export_dir='./images', ext='jpeg'):

        print('image_dir: {}'.format(image_dir))
        print('export_dir: {}'.format(export_dir))

        if os.path.exists(image_dir) and os.path.exists(export_dir):

            filenames = get_file_list(image_dir, ext=ext)
            print('There are {} images to be exported.'.format(len(filenames)))

            for filename in filenames:

                print('Processing {}'.format(filename))
                image_path = os.path.join(image_dir, filename)
                export_path = os.path.join(export_dir, filename)

                image = Image.open(image_path)
                self.draw_image(image)
                image.save(export_path)

        else:
            print("Error: please the path of directory.")

