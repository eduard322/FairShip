import ROOT
import shipDet_conf
import rootUtils as ut
from ShipGeoConfig import ConfigRegistry
from rootpyPickler import Unpickler
from array import array


class SciFiMapping:
	def __init__(self, modules):
		self.modules = modules
		self.scifi = modules['MTC']


	def getFibre2SiPMCPP(self):
		FU, FV = self.scifi.GetSiPMmapU(), self.scifi.GetSiPMmapV()
		self.fibresSiPMU, self.fibresSiPMV  = {}, {}
		for x1,x2 in zip(FU, FV):
			self.fibresSiPMU[x1.first]={}
			for z in x1.second:
				self.fibresSiPMU[x1.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
			self.fibresSiPMV[x2.first]={}
			for z in x2.second:
				self.fibresSiPMV[x2.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}


	def getSiPM2FibreCPP(self):
		XU, XV = self.scifi.GetFibresMapU(), self.scifi.GetFibresMapV()
		self.siPMFibresU, self.siPMFibresV = {}, {}
		for x1, x2 in zip(XU, XV):
			self.siPMFibresU[x1.first]={}
			for z in x1.second:
				self.siPMFibresU[x1.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
			self.siPMFibresV[x2.first]={}
			for z in x2.second:
				self.siPMFibresV[x2.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}

	def make_mapping(self):
		self.scifi.SiPMOverlap()
		self.scifi.SiPMmapping()
		self.getFibre2SiPMCPP()
		self.getSiPM2FibreCPP()

	def get_siPMFibres(self):
		return self.siPMFibresU, self.siPMFibresV


if __name__ == "__main__":
	print("open geofile")
	geoFile = "geofile_full.PG_13-TGeant4.root"
	fgeo = ROOT.TFile.Open(geoFile)

#load geo dictionary
	upkl    = Unpickler(fgeo)
	ship_geo = upkl.load('ShipGeo')

# -----Create geometry----------------------------------------------

	run = ROOT.FairRunSim()
	run.SetName("TGeant4")  # Transport engine
	run.SetSink(ROOT.FairRootFileSink(ROOT.TMemFile('output', 'recreate')))  # Output file
	run.SetUserConfig("g4Config_basic.C") # geant4 transport not used
	rtdb = run.GetRuntimeDb()
	modules = shipDet_conf.configure(run,ship_geo)
	sGeo = fgeo.FAIRGeom
	top = sGeo.GetTopVolume()

	SciFiMapping = SciFiMapping(modules)
	SciFiMapping.make_mapping()

# get mapping from C++

# def getFibre2SiPMCPP(scifi):
# 	FU, FV = scifi.GetSiPMmapU(), scifi.GetSiPMmapV()
# 	fibresSiPMU, fibresSiPMV  = {}, {}
# 	for x1,x2 in zip(FU, FV):
# 		fibresSiPMU[x1.first]={}
# 		for z in x1.second:
# 			fibresSiPMU[x1.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
# 		fibresSiPMV[x2.first]={}
# 		for z in x2.second:
# 			fibresSiPMV[x2.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
# 	return fibresSiPMU, fibresSiPMV


# def getSiPM2FibreCPP(scifi):
# 	XU, XV = scifi.GetFibresMapU(), scifi.GetFibresMapV()
# 	siPMFibresU, siPMFibresV = {}, {}
# 	for x1, x2 in zip(XU, XV):
# 		siPMFibresU[x1.first]={}
# 		for z in x1.second:
# 			siPMFibresU[x1.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
# 		siPMFibresV[x2.first]={}
# 		for z in x2.second:
# 			siPMFibresV[x2.first][z.first]={'weight':z.second[0],'xpos':z.second[1]}
# 	return siPMFibresU, siPMFibresV


# def localPosition(fibresSiPM):
#    meanPos = {}
#    for N in fibresSiPM:
#         m = 0
#         w = 0
#         for fID in fibresSiPM[N]:
#             m+=fibresSiPM[N][fID]['weight']*fibresSiPM[N][fID]['xpos']
#             w+=fibresSiPM[N][fID]['weight']
#         meanPos[N] = m/w
#    return meanPos

# h={}
# def overlap(F,Finv):
#    ut.bookHist(h,'W','overlap/fibre',110,0.,1.1)
#    ut.bookHist(h,'C','coverage',50,0.,10.)
#    ut.bookHist(h,'S','fibres/sipm',15,-0.5,14.5)
#    ut.bookHist(h,'Eff','efficiency',110,0.,1.1)
#    for n in F:
#      C=0
#      for fid in F[n]:
#           rc = h['W'].Fill(F[n][fid]['weight'])
#           C+=F[n][fid]['weight']
#      rc = h['C'].Fill(C)
#    for n in Finv:
#      E=0
#      for sipm in Finv[n]:
#           E+=Finv[n][sipm]['weight']
#      rc = h['Eff'].Fill(E)
#      rc = h['S'].Fill(len(Finv[n]))

# def test(chan):
#     for x in F['VertMatVolume'][chan]:
#             print('%6i:%5.2F'%(x,F['VertMatVolume'][chan][x]))

# def inspectSiPM():
# 	sGeo = ROOT.gGeoManager
# 	SiPMmapVol = sGeo.FindVolumeFast("SiPMmapVol")
# 	for l1 in  SiPMmapVol.GetNodes():   # 3 mats with 4 SiPMs each and 128 channels
# 		t1 = l1.GetMatrix().GetTranslation()[1]
# 		if l1.GetNumber()%1000==0: print( "%s  %5.2F"%(l1.GetName(),(t1)*10.))

# def inspectMats():
# 	sGeo = ROOT.gGeoManager
# 	ScifiHorPlaneVol = sGeo.FindVolumeFast("ScifiHorPlaneVol")
# 	for l1 in  ScifiHorPlaneVol.GetNodes():    # 3 mats
# 		t1 = l1.GetMatrix().GetTranslation()[1]
# 		print(l1.GetName(),t1)
# 		for l2 in  l1.GetVolume().GetNodes():
# 			t2 = l2.GetMatrix().GetTranslation()[1]
# 			print("       ",l2.GetName(),t2,t1+t2)
# def checkFibreCoverage(Finv):
# 	for mat in range(1,4):
# 		for row in range(1,7):
# 			for channel in range(1,473):
# 				fID = mat*10000+row*1000+channel
# 				if not fID in Finv: print('missing fibre:',fID)
# def checkLocalPosition(fibresSiPM):
# 	ut.bookHist(h,'delta','central - weighted',100,-20.,20.)
# 	L = localPosition(fibresSiPM)
# 	sGeo = ROOT.gGeoManager
# 	SiPMmapVol = sGeo.FindVolumeFast("SiPMmapVol")
# 	for l in  SiPMmapVol.GetNodes():
# 		n  = l.GetNumber()
# 		ty = l.GetMatrix().GetTranslation()[1]
# 		delta = ty-L[n]
# 		rc = h['delta'].Fill(delta*10*1000.)
# def moreChecks(modules):
# 	ut.bookHist(h,'dx','dx',100,-0.1,0.1)
# 	ut.bookHist(h,'dy','dy',100,-1.,1.)
# 	ut.bookHist(h,'dz','dz',100,-0.2,0.2)
# 	AS=ROOT.TVector3()
# 	BS=ROOT.TVector3()
# 	AF=ROOT.TVector3()
# 	BF=ROOT.TVector3()
# 	scifi   = modules['Scifi']
# 	F=getFibre2SiPMCPP(modules)
# 	for station in range(1,6):
# 		for orientation in range(0,2):
# 			for channel in F:
# 				globChannel = station* 1000000+ orientation*100000 +channel
# 				scifi.GetSiPMPosition(globChannel,AS,BS)
# 				for fibre in F[channel]:
# 					globFibre = station* 1000000+ orientation*100000+fibre
# 					scifi.GetPosition(globFibre,AF,BF)
# 					if orientation==0: 
# 						dx = AS[1]-AF[1]
# 						dy = AS[0]-AF[0]
# 					else: 
# 						dx = AS[0]-AF[0]
# 						dy = AS[1]-AF[1]
# 					dz = AS[2]-AF[2]
# 					rc = h['dx'].Fill(dx)
# 					rc = h['dy'].Fill(dy)
# 					rc = h['dz'].Fill(dz)
# def someDrawings(F,channel):
#    AF = ROOT.TVector3()
#    BF = ROOT.TVector3()
#    ut.bookCanvas(h,'c1',' ;x;y',800,800,1,1)
#    s = int(channel/1000000)
#    o = int( (channel-1000000*s)/100000)
#    locChannel = channel%100000
#    fibreVol = sGeo.FindVolumeFast('FiberVolume')
#    R = fibreVol.GetShape().GetDX()
#    sipmVol = sGeo.FindVolumeFast("ChannelVol")
#    DY = sipmVol.GetShape().GetDY()
#    DZ = sipmVol.GetShape().GetDZ()
#    n = 0
#    xmin = 999.
#    xmax = -999.
#    ymin = 999.
#    ymax = -999.
#    for fibre in F[locChannel]:
#       globFibre = int(s*1000000 + o*100000 + fibre)
#       scifi.GetPosition(globFibre,AF,BF)
#       loc = scifi.GetLocalPos(globFibre,AF)
#       h['ellipse'+str(n)]=ROOT.TEllipse(loc[0],loc[2],R,0)
#       n+=1
#       if xmin>loc[0]: xmin = loc[0]
#       if ymin>loc[2]: ymin = loc[2]
#       if xmax<loc[0]: xmax = loc[0]
#       if ymax<loc[2]: ymax = loc[2]
#    print(xmin,xmax,ymin,ymax)
#    D = ymax - ymin+3*R
#    x0 = (xmax+xmin)/2.
#    ut.bookHist(h,'x','',100,x0-D/2,x0+D/2,100,ymin-1.5*R,ymax+1.5*R)
#    h['x'].SetStats(0)
#    h['x'].Draw()
#    for i in range(n):
#       print(fibre,globFibre,loc[0],loc[1],loc[2])
#       el = h['ellipse'+str(i)]
#       el.SetFillColor(6)
#       el.Draw('same')
#    scifi.GetSiPMPosition(channel,AF,BF)
#    loc = scifi.GetLocalPos(globFibre,AF)
#    h['rectang']=ROOT.TBox(loc[0]-DY,loc[2]-DZ,loc[0]+DY,loc[2]+DZ)
#    h['rectang'].SetFillColor(4)
#    h['rectang'].SetFillStyle(3001)
#    h['rectang'].Draw('same')
