import cv2
import numpy as np
from Cfg import CCfg
from CamCal import CCamCal, C2dPtSel

def main():
    oImgFrm = None
    # oCamInParam = None
    # ovDistCoeff = None
    oCamCal = CCamCal()
    oCfg = CCfg()

    # read configuration file
    # if len(argv) > 2:
    #     print("usage:", argv[0], "<cfg_file_path>")
    #     return 0
    # elif len(argv) == 2:
    #     oCfg.ldCfgFl(argv[1])
    # else:
    #     oCfg.ldCfgFl(None)
    oCfg.ldCfgFl(None)

    # read frame image
    oImgFrm = cv2.imread(oCfg.m_acInFrmPth, cv2.IMREAD_COLOR)

    # Add 3D image
    oImgPic0 = cv2.imread(oCfg.m_acPic0, cv2.IMREAD_COLOR)

    # resize frame if necessary
    if oCfg.m_nRszFrmHei > 0:
        oFrmSz2d = ((oImgFrm.shape[1] / oImgFrm.shape[0]) * oCfg.m_nRszFrmHei, oCfg.m_nRszFrmHei)
        oImgFrm = cv2.resize(oImgFrm, (int(oFrmSz2d[0]), int(oFrmSz2d[1])))
        oImgPic0 = cv2.resize(oImgPic0, (int(oFrmSz2d[0]), int(oFrmSz2d[1])))
        

    # correct camera distortion
    if oCfg.m_bCalDistFlg:
        oImgUndist = cv2.undistort(oImgFrm, oCfg.m_oCalIntMat, oCfg.m_oCalDistCoeffMat)
        oImgFrm = oImgUndist.copy()

    # initialize the camera calibrator
    oCamCal.initialize(oCfg, oImgFrm, oImgPic0)

    # run camera calibration
    oCamCal.process()

    # output calibration results
    oCamCal.output()

if __name__ == "__main__":
    main()
