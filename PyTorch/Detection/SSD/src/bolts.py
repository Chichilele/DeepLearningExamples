import torch

import os


class FTIBoltDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_folder,
        is_training=False,
        transform=None,
        neg_samples=None,
        transform_neg=None,
    ):
        self.data_folder = data_folder
        self.images_dir = os.path.join(self.data_folder, "images")
        self.annotation_dir = os.path.join(self.data_folder, "annotations")
        self.is_training = is_training
        self.transform = transform
        self.transform_neg = transform_neg
        data = mp.get_xmlstream_from_dir(
            self.annotation_dir,
            list_fields=["object"],
            flatten_fields=["bndbox", "size"],
            skip_fields=["pose", "source", "path"],
        )
        self.annotations = [annotation for annotation in data]
        self.negatives = []
        if neg_samples is not None:
            self.negatives = [
                os.path.join(neg_samples, sample) for sample in os.listdir(neg_samples)
            ]

    def __getitem__(self, idx):

        filename = self.annotations[idx]["filename"]
        img = cv2.imread(os.path.join(self.images_dir, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objects = self.annotations[idx]["object"]

        num_objs = len(objects)
        boxes = []
        for obj in objects:
            xmin = float(obj["bndbox_xmin"])  # left
            xmax = float(obj["bndbox_xmax"])  # right
            ymin = float(obj["bndbox_ymin"])  # bottom
            ymax = float(obj["bndbox_ymax"])  # top
            boxes.append([xmin, ymin, xmax, ymax])

        if self.transform is not None:
            boxes = [[box[0], box[1], box[2], box[3], "bolt"] for box in boxes]
            transformed = self.transform(image=img, bboxes=boxes)
            img = transformed["image"]
            boxes = [box[:-1] for box in transformed["bboxes"]]
        img = Image.fromarray(img)
        trans = T.ToTensor()

        img = trans(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # to fix
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        if len(boxes) != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.is_training is not True:
            # validation
            target["domain"] = torch.tensor([1])  # 1 -> real
        else:
            target["domain"] = torch.tensor([0])  # 0 -> synthetic

        return img, target

    def __len__(self):
        return len(self.annotations)
