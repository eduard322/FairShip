#ifndef MtcDetPoint_H_
#define MtcDetPoint_H_ 1

#include "FairMCPoint.h"
#include "TObject.h"
#include "TVector3.h"

class MtcDetPoint : public FairMCPoint
{

  public:
    /** Default constructor **/
    MtcDetPoint();

    /** Constructor with arguments
     *@param trackID  Index of MCTrack
     *@param detID    Detector ID
     *@param pos      Ccoordinates at entrance to active volume [cm]
     *@param mom      Momentum of track at entrance [GeV]
     *@param tof      Time since event start [ns]
     *@param length   Track length since creation [cm]
     *@param eLoss    Energy deposit [GeV]
     **/

    MtcDetPoint(Int_t trackID,
                Int_t detID,
                TVector3 pos,
                TVector3 mom,
                Double_t tof,
                Double_t length,
                Double_t eLoss,
                Int_t pdgcode);

    /** Destructor **/
    virtual ~MtcDetPoint();

    /** Output to screen **/
    virtual void Print() const;
    Int_t PdgCode() const { return fPdgCode; }
    // Float_t GetEnergyLoss() const { return fEnergyLoss; }
    // Float_t GetX() const { return fX; }
    // Float_t GetY() const { return fY; }
    // Float_t GetZ() const { return fZ; }
    // Float_t GetTime() const { return fTime; }
    // Int_t GetDetectorID() const { return fDetectorID; }
    Int_t GetStationType() const { return int(fDetectorID / 100000) % 10; }
    Int_t GetLayer();
    Int_t GetLayerType();
    // /** Copy constructor **/
    Int_t fPdgCode;
    // Float_t fEnergyLoss; // Energy loss in keV
    // Float_t fX;         // X position in cm
    // Float_t fY;         // Y position in cm
    // Float_t fZ;         // Z position in cm
    // Float_t fTime;      // Time in ns
    // Int_t fDetectorID;  // Detector ID

    MtcDetPoint(const MtcDetPoint& point);
    MtcDetPoint operator=(const MtcDetPoint& point);

    ClassDef(MtcDetPoint, 2)
};

#endif   // MTCDET_MtcDetPoint_H_
