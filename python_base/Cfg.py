import json
import cv2
import numpy as np

class CCfg:
    def __init__(self):
        self.m_acInFrmPth = "./data/frm.jpg"
        self.m_acOutCamMatPth = "./data/calibration.txt"
        self.m_acOutCalDispPth = "./data/calibration.jpg"
        self.m_bOutCalDispFlg = False
        self.m_nRszFrmHei = -1
        self.m_bCalSel2dPtFlg = True
        self.m_voCal2dPt = []
        self.m_voCal3dPt = []
        self.m_nCalTyp = 0
        self.m_fCalRansacReprojThld = 3.0
        self.m_oCalDispGrdDim = (10, 10)
        self.m_bCalDistFlg = False
        self.m_vfCalDistCoeff = []
        self.m_vfCalFocLen = []
        self.m_vfCalPrinPt = []
        self.m_oCalIntMat = np.eye(3, 3, dtype=np.float64)
        self.m_oCalDistCoeffMat = np.zeros((4, 1), dtype=np.float64)

    def ldCfgFl(self, acCfgFlPth=None):
        if acCfgFlPth is None:
            acCfgFlPth = "../data/cfg.json"

        with open(acCfgFlPth, "r") as poCfgFl:
            config_data = json.load(poCfgFl)

        # 3D graph
        self.m_acPic0 = config_data['genInfo'].get("pic0", "../data/pic0.jpg")
        self.m_acInFrmPth = config_data['genInfo'].get("inFrmPth", "../data/frm.jpg")
        self.m_acOutCamMatPth = config_data['genInfo'].get("outCamMatPth", "../data/calibration.txt")
        self.m_acOutCalDispPth = config_data['genInfo'].get("outCalDispPth", "../data/calibration.jpg")
        self.m_bOutCalDispFlg = config_data['genInfo'].get("outCalDispFlg", False)
        self.m_nRszFrmHei = config_data['genInfo'].get("rszFrmHei", -1)
        self.m_bCalSel2dPtFlg = config_data['camCal'].get("calSel2dPtFlg", True)
        self.m_voCal2dPt = config_data['camCal'].get("cal2dPtLs", [])
        self.m_voCal3dPt = config_data['camCal'].get("cal3dPtLs", [])
        self.m_nCalTyp = config_data['camCal'].get("calTyp", 0)
        self.m_fCalRansacReprojThld = config_data['camCal'].get("calRansacReprojThld", 3.0)
        self.m_oCalDispGrdDim = tuple(config_data['camCal'].get("calDispGrdDim", [10, 10]))
        self.m_bCalDistFlg = config_data['camCal'].get("calDistFlg", False)
        self.m_vfCalDistCoeff = config_data['camCal'].get("calDistCoeff", [])
        self.m_vfCalFocLen = config_data['camCal'].get("calFocLen", [])
        self.m_vfCalPrinPt = config_data['camCal'].get("calPrinPt", [])

        # Assertion

        # assert len(self.m_voCal3dPt) >= 4
        # if not self.m_bCalSel2dPtFlg:
        #     assert len(self.m_voCal2dPt) == len(self.m_voCal3dPt)
        assert self.m_nCalTyp in [0, 4, 8, -1]
        assert self.m_fCalRansacReprojThld >= 1
        assert all(dim >= 1 for dim in self.m_oCalDispGrdDim)

        if self.m_bCalDistFlg:
            assert len(self.m_vfCalDistCoeff) == 4
            assert len(self.m_vfCalFocLen) in [1, 2]
            assert len(self.m_vfCalPrinPt) == 2

            self.m_oCalDistCoeffMat[:4, 0] = self.m_vfCalDistCoeff

            if len(self.m_vfCalFocLen) == 1:
                self.m_oCalIntMat[0, 0] = self.m_vfCalFocLen[0]
                self.m_oCalIntMat[1, 1] = self.m_vfCalFocLen[0]
            elif len(self.m_vfCalFocLen) == 2:
                self.m_oCalIntMat[0, 0] = self.m_vfCalFocLen[0]
                self.m_oCalIntMat[1, 1] = self.m_vfCalFocLen[1]

            self.m_oCalIntMat[0, 2] = self.m_vfCalPrinPt[0]
            self.m_oCalIntMat[1, 2] = self.m_vfCalPrinPt[1]

if __name__ == "__main__":
    cfg = CCfg()
    cfg.ldCfgFl()
