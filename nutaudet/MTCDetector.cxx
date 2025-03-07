#include "MTCDetector.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoMedium.h"
#include "TGeoUniformMagField.h"
#include "FairRootManager.h"

#include "TGeoBBox.h"
#include "TGeoTrd1.h"
#include "TGeoCompositeShape.h"
#include "TGeoTube.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TParticle.h"
#include "TVector3.h"

#include "FairVolume.h"
#include "FairGeoVolume.h"
#include "FairGeoNode.h"
#include "FairRootManager.h"
#include "FairGeoLoader.h"
#include "FairGeoInterface.h"
#include "FairGeoMedia.h"
#include "FairGeoBuilder.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"

#include "ShipDetectorList.h"
#include "ShipUnit.h"
#include "ShipStack.h"

#include "TGeoUniformMagField.h"
#include <stddef.h>                     // for NULL
#include <iostream>                     // for operator<<, basic_ostream, etc


TGeoVolume* CreateSegmentedLayer(const char* name, Double_t width, Double_t height,
                                Double_t thickness, Double_t cellSizeX, Double_t cellSizeY,
                                TGeoMedium* material, Int_t color, Double_t transparency) {
    TGeoBBox* mother = new TGeoBBox(Form("%s_mother", name), width/2, height/2, thickness/2);
    TGeoVolume* motherVol = new TGeoVolume(Form("%s_mother", name), mother, material);
    motherVol->SetLineColor(color);
    motherVol->SetTransparency(transparency);

    TGeoBBox* cell = new TGeoBBox(Form("%s_cell", name), cellSizeX/2, cellSizeY/2, thickness/2);
    TGeoVolume* cellVol = new TGeoVolume(Form("%s_cell", name), cell, material);
    cellVol->SetLineColor(color);
    cellVol->SetTransparency(transparency);

    Int_t nX = Int_t(width/cellSizeX);
    Int_t nY = Int_t(height/cellSizeY);
    
    for(Int_t i=0; i<nX; i++) {
        for(Int_t j=0; j<nY; j++) {
            Double_t x = -width/2 + cellSizeX*(i+0.5);
            Double_t y = -height/2 + cellSizeY*(j+0.5);
            motherVol->AddNode(cellVol, i*nY+j, new TGeoTranslation(x, y, 0));
        }
    }
    return motherVol;
}

MTCDetector::MTCDetector(const char* name, Double_t zCenter, Bool_t Active, const char* Title, Int_t DetId)
    : FairDetector(name, Active, DetId),
      fWidth(0), fHeight(0), fIronThick(0), fSciFiThick(0), fScintThick(0),
      fLayers(0), fZCenter(zCenter), fFieldY(0), fMTCDetectorPointCollection(nullptr) {}

MTCDetector::~MTCDetector() {
    if(fMTCDetectorPointCollection) {
        fMTCDetectorPointCollection->Delete();
        delete fMTCDetectorPointCollection;
    }
}

// -----   Private method InitMedium
Int_t MTCDetector::InitMedium(const char* name)
{
    static FairGeoLoader *geoLoad=FairGeoLoader::Instance();
    static FairGeoInterface *geoFace=geoLoad->getGeoInterface();
    static FairGeoMedia *media=geoFace->getMedia();
    static FairGeoBuilder *geoBuild=geoLoad->getGeoBuilder();

    FairGeoMedium *ShipMedium=media->getMedium(name);

    if (!ShipMedium)
    {
        Fatal("InitMedium","Material %s not defined in media file.", name);
        return -1111;
    }
    TGeoMedium* medium=gGeoManager->GetMedium(name);
    if (medium!=NULL)
        return ShipMedium->getMediumIndex();
    return geoBuild->createMedium(ShipMedium);
}

void MTCDetector::SetMTCParameters(Double_t w, Double_t h, Double_t iron, 
                                  Double_t sciFi, Double_t scint, Int_t layers,
                                  Double_t z, Double_t field) {
    fWidth = w;
    fHeight = h;
    fIronThick = iron;
    fSciFiThick = sciFi;
    fScintThick = scint;
    fLayers = layers;
    fZCenter = z;
    fFieldY = field;
}


// Helper function to build the composite SciFi module (fiber module)
TGeoVolume* CreateSciFiModule(const char* name, Double_t width, Double_t height, Double_t thickness) {
    // Here we follow the GEANT4 substructure:
    //   Lower internal iron: 3 mm  (0.3 cm)
    //   Fiber Mat 1 (U):      1.35 mm (0.135 cm)
    //   Air Gap:              1 mm  (0.1 cm)
    //   Fiber Mat 2 (V):      1.35 mm (0.135 cm)
    //   Upper internal iron:  3 mm  (0.3 cm)
    // Total = 0.3 + 0.135 + 0.1 + 0.135 + 0.3 = 1.0 cm (which should equal thickness)
  
    Double_t lowerIronThick = 0.3;
    Double_t fiberMatThick  = 0.135;
    Double_t airGap         = 0.1;
    Double_t upperIronThick = 0.3;
  
    // Mother volume for the SciFi module
    TGeoBBox* modMother = new TGeoBBox(Form("%s_mother", name), width/2, height/2, thickness/2);
    // We use the SciFi material for the mother; adjust if you have a dedicated one
    TGeoVolume* modMotherVol = new TGeoVolume(Form("%s_mother", name), modMother, gGeoManager->GetMedium("SciFiMat"));
    modMotherVol->SetLineColor(kGreen+2);
    modMotherVol->SetTransparency(40);
  
    // --- Lower Internal Iron ---
    TGeoBBox* lowerIronBox = new TGeoBBox(Form("%s_lowerIron", name), width/2, height/2, lowerIronThick/2);
    TGeoVolume* lowerIronVol = new TGeoVolume(Form("%s_lowerIron", name), lowerIronBox, gGeoManager->GetMedium("iron"));
    lowerIronVol->SetLineColor(kGray+1);
    lowerIronVol->SetTransparency(20);
    // Position: at the bottom of the module
    modMotherVol->AddNode(lowerIronVol, 1, new TGeoTranslation(0, 0, -thickness/2 + lowerIronThick/2));
  
    // --- Fiber Mat U ---
    TGeoBBox* fiberMatBoxU = new TGeoBBox(Form("%s_fiberMat_U", name), width/2, height/2, fiberMatThick/2);
    TGeoVolume* fiberMatVolU = new TGeoVolume(Form("%s_fiberMat_U", name), fiberMatBoxU, gGeoManager->GetMedium("SciFiMat"));
    fiberMatVolU->SetLineColor(kYellow);
    fiberMatVolU->SetTransparency(30);
    // Position: above lower iron
    modMotherVol->AddNode(fiberMatVolU, 1, new TGeoTranslation(0, 0, -thickness/2 + lowerIronThick + fiberMatThick/2));
  
    // --- Fiber Mat V ---
    TGeoBBox* fiberMatBoxV = new TGeoBBox(Form("%s_fiberMat_V", name), width/2, height/2, fiberMatThick/2);
    TGeoVolume* fiberMatVolV = new TGeoVolume(Form("%s_fiberMat_V", name), fiberMatBoxV, gGeoManager->GetMedium("SciFiMat"));
    fiberMatVolV->SetLineColor(kYellow);
    fiberMatVolV->SetTransparency(30);
    // Position: above Fiber Mat U plus air gap
    modMotherVol->AddNode(fiberMatVolV, 1, new TGeoTranslation(0, 0, -thickness/2 + lowerIronThick + fiberMatThick + airGap + fiberMatThick/2));
  
    // --- Upper Internal Iron ---
    TGeoBBox* upperIronBox = new TGeoBBox(Form("%s_upperIron", name), width/2, height/2, upperIronThick/2);
    TGeoVolume* upperIronVol = new TGeoVolume(Form("%s_upperIron", name), upperIronBox, gGeoManager->GetMedium("iron"));
    upperIronVol->SetLineColor(kGray+1);
    upperIronVol->SetTransparency(20);
    // Position: at the top of the module
    modMotherVol->AddNode(upperIronVol, 1, new TGeoTranslation(0, 0, thickness/2 - upperIronThick/2));
  
    // Optionally, you can add fiber placements inside the fiber mats here.
  
    return modMotherVol;
  }

void MTCDetector::ConstructGeometry() {
    // Initialize media (using FairROOT’s interface)
    InitMedium("SciFiMat");
    TGeoMedium* SciFiMat = gGeoManager->GetMedium("SciFiMat");
    TGeoMedium* air      = gGeoManager->GetMedium("air");
    TGeoMedium* ironMed  = gGeoManager->GetMedium("iron");
    // For the scintillator, you may use the same medium as SciFiMat or another if defined.
    TGeoMedium* scintMed = gGeoManager->GetMedium("SciFiMat"); 
  
    // Define the module spacing based on three sublayers:
    //   fIronThick (outer iron), fSciFiThick (SciFi module/fiber module), fScintThick (scintillator)
    Double_t moduleSpacing = fIronThick + fSciFiThick + fScintThick;
    Double_t totalLength   = fLayers * moduleSpacing;
  
    // --- Create an envelope volume for the detector (green, semi-transparent) ---
    TGeoBBox* envBox = new TGeoBBox("MTC_env", fWidth/2, fHeight/2, totalLength/2);
    TGeoVolume* envVol = new TGeoVolume("MTC", envBox, air);
    envVol->SetLineColor(kGreen);
    envVol->SetTransparency(50);
  
    // --- Outer Iron Layer (gray) ---
    TGeoBBox* ironBox = new TGeoBBox("MTC_iron", fWidth/2, fHeight/2, fIronThick/2);
    TGeoVolume* ironVol = new TGeoVolume("MTC_iron", ironBox, ironMed);
    ironVol->SetLineColor(kGray+1);
    ironVol->SetTransparency(20);
    // (Optional: attach a magnetic field with TGeoUniformMagField if needed)
  
    // --- SciFi Module ---
    TGeoVolume* sciFiModuleVol = CreateSciFiModule("MTC_sciFi", fWidth, fHeight, fSciFiThick);
  
    // --- Scintillator Layer (blue) ---
    TGeoVolume* scintVol = CreateSegmentedLayer("MTC_scint", fWidth, fHeight,
                                                fScintThick, 1.0, 1.0,
                                                scintMed, kAzure+7, 30);
  
    // --- Assemble the layers into the envelope ---
    for (Int_t i = 0; i < fLayers; i++) {
      // Compute the center position (z) for the current module
      Double_t zPos = -totalLength/2 + (i+0.5) * moduleSpacing;
      // Place the Outer Iron layer (shifted down by half the SciFi+scint thickness)
      envVol->AddNode(ironVol, i, new TGeoTranslation(0, 0, zPos - (fSciFiThick + fScintThick)/2));
      // Place the SciFi module (fiber module)
      envVol->AddNode(sciFiModuleVol, i, new TGeoTranslation(0, 0, zPos - fScintThick/2));
      // Place the Scintillator layer (shifted up by half the iron thickness)
      envVol->AddNode(scintVol, i, new TGeoTranslation(0, 0, zPos + fIronThick/2));
    }
  
    // Finally, add the envelope to the top volume with the global z offset fZCenter
    gGeoManager->GetTopVolume()->AddNode(envVol, 1, new TGeoTranslation(0, 0, fZCenter));
}
// Standard FairDetector methods
void MTCDetector::Initialize() { FairDetector::Initialize(); }


Bool_t  MTCDetector::ProcessHits(FairVolume* vol)
{
  /** This method is called from the MC stepping */
  //Set parameters at entrance of volume. Reset ELoss.
  if ( gMC->IsTrackEntering() ) {
    fELoss  = 0.;
    fTime   = gMC->TrackTime() * 1.0e09;
    fLength = gMC->TrackLength();
    gMC->TrackPosition(fPos);
    gMC->TrackMomentum(fMom);
  }
  // Sum energy loss for all steps in the active volume
  fELoss += gMC->Edep();

  // Create vetoPoint when exiting active volume
  if ( gMC->IsTrackExiting()    ||
       gMC->IsTrackStop()       || 
       gMC->IsTrackDisappeared()   ) {

       // if (fELoss == 0. ) { return kFALSE; } // if you do not want hits with zero eloss

       TParticle* p = gMC->GetStack()->GetCurrentTrack();
       Int_t pdgCode = p->GetPdgCode();
       if (!(fOnlyMuons && TMath::Abs(pdgCode)!=13)){ 
         fTrackID  = gMC->GetStack()->GetCurrentTrackNumber();
         Int_t detID;
         gMC->CurrentVolID(detID);
         TLorentzVector Pos;
         gMC->TrackPosition(Pos);
         TLorentzVector Mom;
         gMC->TrackMomentum(Mom);
         Double_t xmean = (fPos.X()+Pos.X())/2. ;
         Double_t ymean = (fPos.Y()+Pos.Y())/2. ;
         Double_t zmean = (fPos.Z()+Pos.Z())/2. ;
         
         AddHit(fTrackID, detID,
            //TVector3(xmean, ymean, zmean), put entrance and exit instead
              TVector3(fPos.X(), fPos.Y(), fPos.Z()),     // entrance position
              TVector3(fMom.Px(), fMom.Py(), fMom.Pz()),  // entrance momentum
              fTime, fLength, fELoss,pdgCode,
              TVector3(Pos.X(),Pos.Y(),Pos.Z()),          // exit position
              TVector3(Mom.Px(), Mom.Py(), Mom.Pz()) );   // exit momentum
         ShipStack* stack = (ShipStack*) gMC->GetStack();
         stack->AddPoint(kVETO);
       }
  }
  if (fLastDetector) gMC->StopTrack();
  return kTRUE;
}


void MTCDetector::Register(){
    //FairRootManager::Instance()->Register("vetoPoint", "veto",
    //                                    fScoringPlanePointCollection, kTRUE);
    TString name  = fVetoName+"Point";
    TString title = fVetoName;
    FairRootManager::Instance()->Register(name, title, fScoringPlanePointCollection, kTRUE);
    std::cout << this->GetName() << ",  Register() says: registered " << fVetoName <<" collection"<<std::endl;
  }
  
TClonesArray* MTCDetector::GetCollection(Int_t) const { return nullptr; }
void MTCDetector::Reset() { /* Implementation if needed */ }

ClassImp(MTCDetector)