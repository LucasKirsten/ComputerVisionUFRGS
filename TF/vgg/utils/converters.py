class Converters:
    def __init__(self):
        self.class_to_id = {
            "background": 0,
            "erythrocytes": 1,
            "spirochaete": 2
        }
        self.class_to_color = {
            "background": [0, 0, 0],
            "erythrocytes": [255, 0, 0],
            "spirochaete": [255, 255, 0]
        }
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        self.num_classes = len(self.class_to_id)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_class_to_id(self):
        return self.class_to_id

    def get_class_to_color(self):
        return self.class_to_color

    def get_id_to_class(self):
        return self.id_to_class

    def get_num_classes(self):
        return self.num_classes

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std
