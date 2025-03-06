#ifndef MTCDETECTOR_H
#define MTCDETECTOR_H

#include "FairDetector.h"
#include "TGeoMatrix.h"
#include "TClonesArray.h"

class MTCDetector : public FairDetector {
public:
    MTCDetector(const char* name, Double_t zCenter, Bool_t Active, const char* Title = "", Int_t DetId = 0);    MTCDetector();
    virtual ~MTCDetector();

    void SetMTCParameters(Double_t width, Double_t height,
                         Double_t ironThick, Double_t sciFiThick, Double_t scintThick,
                         Int_t nLayers, Double_t zCenter, Double_t fieldY);

    virtual void ConstructGeometry();
    virtual void Initialize();
    virtual Bool_t ProcessHits(FairVolume* vol = 0);
    virtual void Register();
    virtual TClonesArray* GetCollection(Int_t iColl) const;
    virtual void Reset();

private:
    Double_t fWidth;
    Double_t fHeight;
    Double_t fIronThick;
    Double_t fSciFiThick;
    Double_t fScintThick;
    Int_t fLayers;
    Double_t fZCenter;
    Double_t fFieldY;
    TClonesArray* fMTCDetectorPointCollection;

    MTCDetector(const MTCDetector&);
    MTCDetector& operator=(const MTCDetector&);
    Int_t InitMedium(const char* name);
    ClassDef(MTCDetector, 1)
};

#endif