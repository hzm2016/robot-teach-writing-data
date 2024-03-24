from tools import cropping, binarize, rescale, rotate

class Postprocessor(object):

    def __init__(self, args) -> None:

        self.pipeline = []
        for k, v in args.items():
            setattr(self, k.lower(), v)
            self.pipeline.append(k.upper())

    def CROPPING(self, image):

        return cropping(image, self.cropping)

    def BINARIZE(self, image):

        return binarize(image, self.binarize)

    def RESCALE(self, image):

        return rescale(image, self.rescale)

    def ROTATE(self, image):

        return rotate(image, self.rotate)

    def process(self, image):

        for m in self.pipeline:
            image = getattr(self, m)(image)

        return image
