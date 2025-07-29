import ROOT

class PyScifiCluster:
    def __init__(self, first, N, hit_list, scifiDet, withQDC=False):
        # mirror the C++ ctor initialization
        self.fType = 0
        self.fTime = float('inf')
        self.fFirst = first
        self.fN = N
        self.fMeanPositionA = ROOT.TVector3(0, 0, 0)
        self.fMeanPositionB = ROOT.TVector3(0, 0, 0)
        weight = 0.0

        # accumulate weighted positions & find earliest time
        for k in range(N):
            det_id = first + k
            A = ROOT.TVector3()
            B = ROOT.TVector3()
            scifiDet.GetSiPMPosition(det_id, A, B)

            w = 1.0
            if withQDC:
                w = hit_list[k].GetEnergy()

            t = 6.25 * hit_list[k].GetTime()
            weight += w

            # weighted sum
            self.fMeanPositionA += A * w
            self.fMeanPositionB += B * w

            # earliest time
            if t < self.fTime:
                self.fTime = t

        # finalize mean positions
        if weight > 0.0:
            inv = 1.0 / weight
            self.fMeanPositionA *= inv
            self.fMeanPositionB *= inv

        self.fEnergy = weight