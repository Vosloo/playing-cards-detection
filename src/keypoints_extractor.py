import imgaug as ia
from imgaug import augmenters as iaa
from shapely.geometry import Polygon


class KeypointExtractor:
    def __init__(self, img_size: tuple, decals: tuple, decals3: tuple) -> None:
        self.imgW, self.imgH = img_size
        self.decalX, self.decalY = decals
        self.decalX3, self.decalY3 = decals3

    def _create_transformations(self):
        # imgaug transformation for one card in scenario with 2 cards
        self.transform_1card = iaa.Sequential(
            [
                iaa.Affine(scale=[0.65, 1]),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)}),
            ]
        )

        # For the 3 cards scenario, we use 3 imgaug transforms, the first 2 are for individual cards,
        # and the third one for the group of 3 cards
        self.trans_rot1 = iaa.Sequential(
            [iaa.Affine(translate_px={"x": (10, 20)}), iaa.Affine(rotate=(22, 30))]
        )
        self.trans_rot2 = iaa.Sequential(
            [iaa.Affine(translate_px={"x": (0, 5)}), iaa.Affine(rotate=(10, 15))]
        )
        self.transform_3cards = iaa.Sequential(
            [
                iaa.Affine(translate_px={"x": self.decalX - self.decalX3, "y": self.decalY - self.decalY3}),
                iaa.Affine(scale=[0.65, 1]),
                iaa.Affine(rotate=(-180, 180)),
                iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            ]
        )

        # imgaug transformation for the background
        self.scaleBg = iaa.Scale({"height": self.imgH, "width": self.imgW})

    def kps_to_polygon(self, kps):
        """
            Convert imgaug keypoints to shapely polygon
        """
        pts = [(kp.x, kp.y) for kp in kps]
        return Polygon(pts)

    def hull_to_kps(self, hull, decalX=None, decalY=None):
        """
            Convert hull to imgaug keypoints
        """
        # hull is a cv2.Contour, shape : Nx1x2
        decalX = self.decalX if decalX is None else decalX
        decalY = self.decalY if decalY is None else decalY

        kps = [
            ia.Keypoint(x=p[0] + decalX, y=p[1] + decalY) for p in hull.reshape(-1, 2)
        ]
        kps = ia.KeypointsOnImage(kps, shape=(self.imgH, self.imgW, 3))
        return kps

    def kps_to_BB(self, kps):
        """
            Determine imgaug bounding box from imgaug keypoints
        """
        extend = 3  # To make the bounding box a little bit bigger
        kpsx = [kp.x for kp in kps.keypoints]
        minx = max(0, int(min(kpsx) - extend))
        maxx = min(self.imgW, int(max(kpsx) + extend))
        kpsy = [kp.y for kp in kps.keypoints]
        miny = max(0, int(min(kpsy) - extend))
        maxy = min(self.imgH, int(max(kpsy) + extend))
        if minx == maxx or miny == maxy:
            return None
        else:
            return ia.BoundingBox(x1=minx, y1=miny, x2=maxx, y2=maxy)
