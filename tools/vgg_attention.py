import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2

def inv_vgg_normalization(image):
  return np.clip(image + [123.68, 116.78, 103.94],0,255)

if __name__=="__main__":
    # 加载VGG16模型
    model = models.vgg16(pretrained=True)
    model.eval()

    # 加载并预处理输入图片
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载你的输入图片
    image = Image.open('your_image.jpg')
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 前向传播直到Conv3_3层
    conv3_3_output = None
    for layer in model.features:
        image = layer(image)
        if 'Conv3' in str(layer):
            conv3_3_output = image

    # 此时conv3_3_output包含了在Conv3_3层的输出
    Image.fromarray(conv3_3_output).save('images/vgg_attn.jpg')