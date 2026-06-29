from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class Matching:
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def match(self, img1_path, img2_path, threshold):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor1 = preprocess(img1).unsqueeze(0)
        img_tensor2 = preprocess(img2).unsqueeze(0)

        img_embedding1 = self.resnet(img_tensor1)
        img_embedding2 = self.resnet(img_tensor2)
        
        cosine_sim = F.cosine_similarity(img_embedding1, img_embedding2)
        return { "status": "SUCCESS",
            "match": (cosine_sim >= threshold).item()}