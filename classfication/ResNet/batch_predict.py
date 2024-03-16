import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet34

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    imgs_root = "./data"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."

    img_path_list = [os.path.join(imgs_root,i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    json_path = "./class_indices.json"

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    model = resnet34(num_classes=5).to(device)
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path),f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path,map_location = device))

    model.eval()
    batch_size = 8

    with torch.no_grad():
        for ids in range(0,len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size : min((ids + 1) * batch_size,len(img_path_list) - 1)]:
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

                batch_img = torch.stack(img_list,dim = 0)
                output = model(batch_img.to(device)).cpu()
                predict = torch.softmax(output, dim=1)
                probs, classes = torch.max(predict, dim=1)
                for idx, (pro, cla) in enumerate(zip(probs, classes)):
                    print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))                
    
if __name__ == '__main__':
    main()