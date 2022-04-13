import torch
import torchvision
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
import PIL.Image as Image
from torchvision import transforms
from Config import Config
config = Config()
model_path = config.model_path
import sys

def predict(model, test_image_list):
    image_transforms = {
        'train':    transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
            # Validation does not use augmentation
        'valid':    transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test':    transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),}
    for test_image_name in test_image_list:
        transform = image_transforms['test']

        test_image = Image.open(test_image_name)
        draw = ImageDraw.Draw(test_image)

        test_image_tensor = transform(test_image)

        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)


        data = {
            'test_images': torchvision.datasets.ImageFolder(root=config.test_path,
                                                            transform=image_transforms['test'])
        }

        idx_to_class = {v: k for k, v in data['test_images'].class_to_idx.items()}
        with torch.no_grad():
            model.eval()
            model = model.type(torch.FloatTensor)
            model = model.cuda()
            out = model(test_image_tensor)
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)
            print("Prediction : ", idx_to_class[topclass.cpu().numpy()[0][0]], ", Score: ", topk.cpu().numpy()[0][0])
            text = idx_to_class[topclass.cpu().numpy()[0][0]] + " " + str(topk.cpu().numpy()[0][0])
            font = ImageFont.truetype('arial.ttf', 36)
            draw.text((0, 0), text, (255, 0, 0), font=font)
            test_image.show()

if __name__ == "__main__":
    model = torch.load(model_path)
    # path = input("please input image path...")
    #for i in range(1, len(sys.argv)):
        #path = sys.argv[1]
    print(sys.argv[1: ])
    predict(model, sys.argv[1: ])
