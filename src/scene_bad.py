import numpy as np

class Scene:
    def __init__(
        self,
        bg,
        img1,
        class1,
        hulla1,
        hullb1,
        img2,
        class2,
        hulla2,
        hullb2,
        img3=None,
        class3=None,
        hulla3=None,
        hullb3=None,
    ):
        if img3 is not None:
            self.create3CardsScene(
                bg,
                img1,
                class1,
                hulla1,
                hullb1,
                img2,
                class2,
                hulla2,
                hullb2,
                img3,
                class3,
                hulla3,
                hullb3,
            )
        else:
            self.create2CardsScene(
                bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2
            )

    def create2CardsScene(
        self, bg, img1, class1, hulla1, hullb1, img2, class2, hulla2, hullb2
    ):
        kpsa1 = hull_to_kps(hulla1)
        kpsb1 = hull_to_kps(hullb1)
        kpsa2 = hull_to_kps(hulla2)
        kpsb2 = hull_to_kps(hullb2)

        # Randomly transform 1st card
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY : decalY + cardH, decalX : decalX + cardW, :] = img1
        self.img1, self.lkps1, self.bbs1 = augment(
            self.img1, [cardKP, kpsa1, kpsb1], transform_1card
        )

        # Randomly transform 2nd card. We want that card 2 does not partially cover a corner of 1 card.
        # If so, we apply a new random transform to card 2
        while True:
            self.listbba = []
            self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
            self.img2[decalY : decalY + cardH, decalX : decalX + cardW, :] = img2
            self.img2, self.lkps2, self.bbs2 = augment(
                self.img2, [cardKP, kpsa2, kpsb2], transform_1card
            )

            # mainPoly2: shapely polygon of card 2
            mainPoly2 = kps_to_polygon(self.lkps2[0].keypoints[0:4])
            invalid = False
            intersect_ratio = 0.1
            for i in range(1, 3):
                # smallPoly1: shapely polygon of one of the hull of card 1
                smallPoly1 = kps_to_polygon(self.lkps1[i].keypoints[:])
                a = smallPoly1.area
                # We calculate area of the intersection of card 1 corner with card 2
                intersect = mainPoly2.intersection(smallPoly1)
                ai = intersect.area
                # If intersection area is small enough, we accept card 2
                if (a - ai) / a > 1 - intersect_ratio:
                    self.listbba.append(BBA(self.bbs1[i - 1], class1))
                # If intersectio area is not small, but also not big enough, we want apply new transform to card 2
                elif (a - ai) / a > intersect_ratio:
                    invalid = True
                    break

            if not invalid:
                break

        self.class1 = class1
        self.class2 = class2
        for bb in self.bbs2:
            self.listbba.append(BBA(bb, class2))
        # Construct final image of the scene by superimposing: bg, img1 and img2
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)

    def create3CardsScene(
        self,
        bg,
        img1,
        class1,
        hulla1,
        hullb1,
        img2,
        class2,
        hulla2,
        hullb2,
        img3,
        class3,
        hulla3,
        hullb3,
    ):

        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(
            self.img3, [cardKP, kpsa3, kpsb3], trans_rot1
        )
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(
            self.img2, [cardKP, kpsa2, kpsb2], trans_rot2
        )
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img3, _lkps3, self.bbs3 = augment(
                self.img3, self.lkps3, det_transform_3cards, False
            )
            if _img3 is None:
                continue
            _img2, _lkps2, self.bbs2 = augment(
                self.img2, self.lkps2, det_transform_3cards, False
            )
            if _img2 is None:
                continue
            _img1, self.lkps1, self.bbs1 = augment(
                self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False
            )
            if _img1 is None:
                continue
            break
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.listbba = [
            BBA(self.bbs1[0], class1),
            BBA(self.bbs2[0], class2),
            BBA(self.bbs3[0], class3),
            BBA(self.bbs3[1], class3),
        ]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)

    def display(self):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(self.final)
        for bb in self.listbba:
            rect = patches.Rectangle(
                (bb.x1, bb.y1),
                bb.x2 - bb.x1,
                bb.y2 - bb.y1,
                linewidth=1,
                edgecolor="b",
                facecolor="none",
            )
            ax.add_patch(rect)

    def res(self):
        return self.final

    def write_files(self, save_dir, display=False):
        jpg_fn, xml_fn = give_me_filename(save_dir, ["jpg", "xml"])
        plt.imsave(jpg_fn, self.final)
        if display:
            print("New image saved in", jpg_fn)
        create_voc_xml(xml_fn, jpg_fn, self.listbba, display=display)

