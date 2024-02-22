import cv2
import numpy as np
from Cfg import CCfg

class C2dPtSel:
    def __init__(self):
        self.m_2dvoNd = []
        self.m_3dvoNd = []
        self.m_oImgFrm = None

    def initialize(self, oCfg, oImgFrm, oImgPic0):
        self.m_oCfg = oCfg
        self.m_oImgFrm = oImgFrm.copy()
        self.m_oImgPic0 = oImgPic0.copy()
        self.bothImg = np.hstack((self.m_oImgFrm, self.m_oImgPic0))
    
    def chk_img_ld(self):
        # TODO
        # if not self.m_oImgFrm:
        #     return False
        # else:
        #     return True
        return True

    def process(self):
        if self.m_oCfg.m_bCalSel2dPtFlg:
            print("Hot keys: \n"
                  "\tESC - exit\n"
                  "\tr - re-select a set of 2D points\n"
                  "\to - finish selecting a set of 2D points\n")

            copy_Img = self.bothImg.copy()

            cv2.namedWindow("selector of 2D and 3D points", cv2.WINDOW_NORMAL)
            cv2.imshow("selector of 2D and 3D points", self.bothImg)
            cv2.setMouseCallback("selector of 2D and 3D points", self.on_mouse)  # Register for mouse event

            while True:
                nKey = cv2.waitKey(0)  # read keyboard event

                if nKey == 27:
                    break

                if nKey == ord('r'):  # reset the nodes
                    self.m_2dvoNd = []
                    self.m_3dvoNd = []
                    self.bothImg = copy_Img.copy()
                    cv2.imshow("selector of 2D and 3D points", self.bothImg)

                if nKey == ord('o'):  # finish selection of pairs of test points
                    cv2.destroyWindow("selector of 2D and 3D points")
                    return self.m_2dvoNd, self.m_3dvoNd

    def on_mouse(self, event, x, y, flags, param):
        if not self.chk_img_ld():
            print("Error: on_mouse(): frame image is unloaded")
            return

        if event == cv2.EVENT_LBUTTONUP and x < self.m_oImgFrm.shape[1]:
            self.add_nd2D(x, y)
        elif event == cv2.EVENT_LBUTTONUP and x >= self.m_oImgFrm.shape[1] and x < self.bothImg.shape[1]:
            self.add_nd3D(x, y)

    def add_nd2D(self, nX, nY):
        oCurrNd = (nX, nY)
        self.m_2dvoNd.append(oCurrNd)
        cv2.circle(self.bothImg, oCurrNd, 6, (255, 0, 0), 1, cv2.LINE_AA)  # draw the circle
        cv2.putText(self.bothImg, str(len(self.m_2dvoNd) - 1), oCurrNd, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("selector of 2D and 3D points", self.bothImg)

    def add_nd3D(self, nX, nY):
        oCurrNd = (nX, nY)
        self.m_3dvoNd.append(oCurrNd)
        cv2.circle(self.bothImg, oCurrNd, 6, (255, 0, 0), 1, cv2.LINE_AA)  # draw the circle
        cv2.putText(self.bothImg, str(len(self.m_3dvoNd) - 1), oCurrNd, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("selector of 2D and 3D points", self.bothImg)


class CCamCal:
    def __init__(self):
        self.m_vo3dPt = []
        self.m_vo2dPt = []
        # self.m_oHomoMat = np.zeros((3, 3), dtype=np.float64)
        # self.m_fReprojErr = np.inf
        self.o2dPtSel = C2dPtSel()

    def initialize(self, oCfg: CCfg, oImgFrm, oImgPic0):
        self.m_oCfg = oCfg
        self.m_oImgFrm = oImgFrm.copy()
        self.m_oImgPic0 = oImgPic0.copy()

        if not self.m_oCfg.m_bCalSel2dPtFlg:
            self.m_vo2dPt = self.m_oCfg.m_voCal2dPt

        self.m_oHomoMat = np.zeros((3, 3), dtype=np.float64)
        self.m_fReprojErr = np.inf

    def process(self):
        #  select 2D points if they are not provided in the configuration file
        if self.m_oCfg.m_bCalSel2dPtFlg:
            self.m_vo2dPt = []
            self.o2dPtSel.initialize(self.m_oCfg, self.m_oImgFrm, self.m_oImgPic0)
            vo2dPt, vo3dPt = self.o2dPtSel.process()

            print("Selected 2D points on the frame image:")
            for i in range(len(vo2dPt)):
                self.m_vo2dPt.append(vo2dPt[i])
                if len(vo2dPt) - 1 > i:
                    print("[ {}, {} ],".format(vo2dPt[i][0], vo2dPt[i][1]))
                else:
                    print("[ {}, {} ]".format(vo2dPt[i][0], vo2dPt[i][1]))

            print("Selected 3D points on the frame image:")
            for i in range(len(vo3dPt)):
                self.m_vo3dPt.append(vo3dPt[i])
                if len(vo3dPt) - 1 > i:
                    print("[ {}, {} ],".format(vo3dPt[i][0], vo3dPt[i][1]))
                else:
                    print("[ {}, {} ]".format(vo3dPt[i][0], vo3dPt[i][1]))

        # compute homography matrix
        if -1 == self.m_oCfg.m_nCalTyp:
            # run all calibration types
            self.runAllCalTyp()
        else:
            self.m_oHomoMat, self.m_fReprojErr = cv2.findHomography(np.array(self.m_vo3dPt), np.array(self.m_vo2dPt),
                                                                     self.m_oCfg.m_nCalTyp, self.m_oCfg.m_fCalRansacReprojThld)
            self.m_fReprojErr = self.calc_reproj_err(self.m_oHomoMat, self.m_oCfg.m_nCalTyp, self.m_oCfg.m_fCalRansacReprojThld)

        print()

    def output(self):
        self.out_txt()
        self.pltDispGrd()

    def runAllCalTyp(self):
        oHomoMat = np.zeros((3, 3), dtype=np.float64)
        fReprojErr = 0

        # a regular method using all the points
        try:
            oHomoMat, fReprojErr = cv2.findHomography(np.array(self.m_vo3dPt), np.array(self.m_vo2dPt), 0, 0)
            fReprojErr = self.calc_reproj_err(oHomoMat, 0, 0)
            if fReprojErr < self.m_fReprojErr:
                self.m_fReprojErr = fReprojErr
                self.m_oHomoMat = oHomoMat
        except cv2.error as e:
            print("Exception caught:", e)

        # Least-Median robust method
        try:
            oHomoMat, fReprojErr = cv2.findHomography(np.array(self.m_vo3dPt), np.array(self.m_vo2dPt), 4, 0)
            fReprojErr = self.calc_reproj_err(oHomoMat, 4, 0)
            if fReprojErr < self.m_fReprojErr:
                self.m_fReprojErr = fReprojErr
                self.m_oHomoMat = oHomoMat
        except cv2.error as e:
            print("Exception caught:", e)

        # RANSAC-based robust method
        for t in range(100, 5, -5):
            try:
                oHomoMat, fReprojErr = cv2.findHomography(np.array(self.m_vo3dPt), np.array(self.m_vo2dPt), 8, t)
                fReprojErr = self.calc_reproj_err(oHomoMat, 8, 0)
                if fReprojErr < self.m_fReprojErr:
                    self.m_fReprojErr = fReprojErr
                    self.m_oHomoMat = oHomoMat
            except cv2.error as e:
                print("Exception caught:", e)


    def calc_reproj_err(self, oHomoMat, nCalTyp, fCalRansacReprojThld):
        fReprojErr = 0

        for i in range(len(self.m_vo3dPt)):
            o3dPtMat = np.array([[self.m_vo3dPt[i][0]], [self.m_vo3dPt[i][1]], [1]], dtype=np.float64)
            o2dPtMat = np.dot(oHomoMat, o3dPtMat)

            o2dPt = (o2dPtMat[0, 0] / o2dPtMat[2, 0], o2dPtMat[1, 0] / o2dPtMat[2, 0])

            fReprojErr += np.linalg.norm(np.array(self.m_vo2dPt[i]) - np.array(o2dPt))

        fReprojErr /= len(self.m_vo3dPt)

        if 8 == nCalTyp:
            print("Average reprojection error of method #{} (threshold: {}): {}".format(nCalTyp, fCalRansacReprojThld, fReprojErr))
        else:
            print("Average reprojection error of method #{}: {}".format(nCalTyp, fReprojErr))

        return fReprojErr

    def out_txt(self):
        with open(self.m_oCfg.m_acOutCamMatPth, "w") as pfHomoMat:
            pfHomoMat.write("Homography matrix: {:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f}\n".format(
                self.m_oHomoMat[0, 0], self.m_oHomoMat[0, 1], self.m_oHomoMat[0, 2],
                self.m_oHomoMat[1, 0], self.m_oHomoMat[1, 1], self.m_oHomoMat[1, 2],
                self.m_oHomoMat[2, 0], self.m_oHomoMat[2, 1], self.m_oHomoMat[2, 2]))
            print("Homography matrix: {:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f}\n".format(
                self.m_oHomoMat[0, 0], self.m_oHomoMat[0, 1], self.m_oHomoMat[0, 2],
                self.m_oHomoMat[1, 0], self.m_oHomoMat[1, 1], self.m_oHomoMat[1, 2],
                self.m_oHomoMat[2, 0], self.m_oHomoMat[2, 1], self.m_oHomoMat[2, 2]))

            if self.m_oCfg.m_bCalDistFlg:
                oCalIntMat = self.m_oCfg.m_oCalIntMat
                pfHomoMat.write("Intrinsic parameter matrix: {:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f}\n".format(
                    oCalIntMat[0, 0], oCalIntMat[0, 1], oCalIntMat[0, 2],
                    oCalIntMat[1, 0], oCalIntMat[1, 1], oCalIntMat[1, 2],
                    oCalIntMat[2, 0], oCalIntMat[2, 1], oCalIntMat[2, 2]))
                print("Intrinsic parameter matrix: {:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f};{:.15f} {:.15f} {:.15f}\n".format(
                    oCalIntMat[0, 0], oCalIntMat[0, 1], oCalIntMat[0, 2],
                    oCalIntMat[1, 0], oCalIntMat[1, 1], oCalIntMat[1, 2],
                    oCalIntMat[2, 0], oCalIntMat[2, 1], oCalIntMat[2, 2]))

                oCalDistCoeffMat = self.m_oCfg.m_oCalDistCoeffMat
                pfHomoMat.write("Distortion coefficients: {:.15f} {:.15f} {:.15f} {:.15f}\n".format(
                    oCalDistCoeffMat[0], oCalDistCoeffMat[1], oCalDistCoeffMat[2], oCalDistCoeffMat[3]))
                print("Distortion coefficients: {:.15f} {:.15f} {:.15f} {:.15f}\n".format(
                    oCalDistCoeffMat[0], oCalDistCoeffMat[1], oCalDistCoeffMat[2], oCalDistCoeffMat[3]))

            pfHomoMat.write("Reprojection error: {:.15f}\n".format(self.m_fReprojErr))
            print("Reprojection error: {:.15f}\n".format(self.m_fReprojErr))

    def pltDispGrd(self):
        oImgPlt = self.m_oImgFrm.copy()
        oDispGrdDim = self.m_oCfg.m_oCalDispGrdDim

        fXMin = float('inf')
        fYMin = float('inf')
        fXMax = -float('inf')
        fYMax = -float('inf')

        for i in range(len(self.m_vo3dPt)):
            if fXMin > self.m_vo3dPt[i][0]:
                fXMin = self.m_vo3dPt[i][0]
            if fYMin > self.m_vo3dPt[i][1]:
                fYMin = self.m_vo3dPt[i][1]
            if fXMax < self.m_vo3dPt[i][0]:
                fXMax = self.m_vo3dPt[i][0]
            if fYMax < self.m_vo3dPt[i][1]:
                fYMax = self.m_vo3dPt[i][1]

        #  compute the endpoints for the 3D grid on the ground plane
        vo3dGrdPtTop, vo3dGrdPtBtm = [], []
        vo3dGrdPtLft, vo3dGrdPtRgt = [], []

        for x in range(oDispGrdDim[0]):
            vo3dGrdPtTop.append(((fXMin + x * ((fXMax - fXMin) / (oDispGrdDim[0] - 1))), fYMin))
            vo3dGrdPtBtm.append(((fXMin + x * ((fXMax - fXMin) / (oDispGrdDim[0] - 1))), fYMax))

        for y in range(oDispGrdDim[1]):
            vo3dGrdPtLft.append((fXMin, (fYMin + y * ((fYMax - fYMin) / (oDispGrdDim[1] - 1)))))
            vo3dGrdPtRgt.append((fXMax, (fYMin + y * ((fYMax - fYMin) / (oDispGrdDim[1] - 1)))))

	    # compute the endpoints for the projected 2D grid
        vo2dGrdPtTop, vo2dGrdPtBtm = [], []
        vo2dGrdPtLft, vo2dGrdPtRgt = [], []

        for i in range(oDispGrdDim[0]):
            o3dPtMat = np.array([[vo3dGrdPtTop[i][0]], [vo3dGrdPtTop[i][1]], [1]])
            o2dPtMat = np.dot(self.m_oHomoMat, o3dPtMat)
            vo2dGrdPtTop.append((o2dPtMat[0, 0] / o2dPtMat[2, 0], o2dPtMat[1, 0] / o2dPtMat[2, 0]))

            o3dPtMat = np.array([[vo3dGrdPtBtm[i][0]], [vo3dGrdPtBtm[i][1]], [1]])
            o2dPtMat = np.dot(self.m_oHomoMat, o3dPtMat)
            vo2dGrdPtBtm.append((o2dPtMat[0, 0] / o2dPtMat[2, 0], o2dPtMat[1, 0] / o2dPtMat[2, 0]))

        for i in range(oDispGrdDim[1]):
            o3dPtMat = np.array([[vo3dGrdPtLft[i][0]], [vo3dGrdPtLft[i][1]], [1]])
            o2dPtMat = np.dot(self.m_oHomoMat, o3dPtMat)
            vo2dGrdPtLft.append((o2dPtMat[0, 0] / o2dPtMat[2, 0], o2dPtMat[1, 0] / o2dPtMat[2, 0]))

            o3dPtMat = np.array([[vo3dGrdPtRgt[i][0]], [vo3dGrdPtRgt[i][1]], [1]])
            o2dPtMat = np.dot(self.m_oHomoMat, o3dPtMat)
            vo2dGrdPtRgt.append((o2dPtMat[0, 0] / o2dPtMat[2, 0], o2dPtMat[1, 0] / o2dPtMat[2, 0]))

	    # draw grid lines on the frame image
        for i in range(oDispGrdDim[0]):
            cv2.line(oImgPlt, tuple(map(int, vo2dGrdPtTop[i])), tuple(map(int, vo2dGrdPtBtm[i])),
                     (int(255.0 * (i / oDispGrdDim[0])), 127, 127), 2, cv2.LINE_AA)

        for i in range(oDispGrdDim[1]):
            cv2.line(oImgPlt, tuple(map(int, vo2dGrdPtLft[i])), tuple(map(int, vo2dGrdPtRgt[i])),
                     (127, 127, int(255.0 * (i / oDispGrdDim[1]))), 2, cv2.LINE_AA)

        # plot the 2D points
        for i, pt in enumerate(self.m_vo2dPt):
            acPtIdx = str(i)
            cv2.circle(oImgPlt, pt, 6, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(oImgPlt, acPtIdx, pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # plot the projected 2D points
		# plot the 2D points
        for i in range(len(self.m_vo2dPt)):
            o3dPtMat = np.array([[self.m_vo3dPt[i][0]], [self.m_vo3dPt[i][1]], [1]])
            o2dPtMat = np.dot(self.m_oHomoMat, o3dPtMat)
            cv2.circle(oImgPlt, (int(o2dPtMat[0, 0] / o2dPtMat[2, 0]), int(o2dPtMat[1, 0] / o2dPtMat[2, 0])),
                       12, (0, 0, 255), 1, cv2.LINE_AA)

        # display plotted image
        cv2.namedWindow("3D grid on the ground plane", cv2.WINDOW_NORMAL)
        cv2.imshow("3D grid on the ground plane", oImgPlt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save plotted image
        if self.m_oCfg.m_bOutCalDispFlg:
            cv2.imwrite(self.m_oCfg.m_acOutCalDispPth, oImgPlt)

