// MTC detector specific headers
#include "MTCDetector.h"

#include "MtcDetPoint.h"
#include "ShipDetectorList.h"
#include "ShipStack.h"
#include "ShipUnit.h"

// ROOT / TGeo headers
#include "TGeoBBox.h"
#include "TGeoCompositeShape.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoPara.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoTube.h"
#include "TGeoUniformMagField.h"
#include "TGeoVolume.h"
#include "TParticle.h"
#include "TVector3.h"

// FairROOT headers
#include "FairGeoBuilder.h"
#include "FairGeoInterface.h"
#include "FairGeoLoader.h"
#include "FairGeoMedia.h"
#include "FairGeoNode.h"
#include "FairGeoVolume.h"
#include "FairRootManager.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairVolume.h"

// Additional standard headers
#include "TClonesArray.h"
#include "TList.h"       // for TListIter, TList (ptr only)
#include "TObjArray.h"   // for TObjArray
#include "TString.h"     // for TString
#include "TVirtualMC.h"

#include <iosfwd>     // for ostream
#include <iostream>   // for operator<<, basic_ostream, etc
#include <stddef.h>   // for NULL
using std::cout;
using std::endl;
using namespace ShipUnit;

MTCDetector::MTCDetector()
    : FairDetector("MTC", kTRUE, kMTC)
    , fTrackID(-1)
    , fPdgCode()
    , fVolumeID(-1)
    , fPos()
    , fMom()
    , fTime(-1.)
    , fLength(-1.)
    , fELoss(-1)
    , fMTCDetectorPointCollection(new TClonesArray("MtcDetPoint"))
{}

MTCDetector::MTCDetector(const char* name, Bool_t Active, const char* Title, Int_t DetId)
    : FairDetector(name, Active, kMTC)
    , fTrackID(-1)
    , fVolumeID(-1)
    , fPos()
    , fMom()
    , fTime(-1.)
    , fLength(-1.)
    , fELoss(-1)
    , fMTCDetectorPointCollection(new TClonesArray("MtcDetPoint"))
{}

MTCDetector::~MTCDetector()
{
    if (fMTCDetectorPointCollection) {
        fMTCDetectorPointCollection->Delete();
        delete fMTCDetectorPointCollection;
    }
}

// -----   Private method InitMedium
Int_t MTCDetector::InitMedium(const char* name)
{
    static FairGeoLoader* geoLoad = FairGeoLoader::Instance();
    static FairGeoInterface* geoFace = geoLoad->getGeoInterface();
    static FairGeoMedia* media = geoFace->getMedia();
    static FairGeoBuilder* geoBuild = geoLoad->getGeoBuilder();

    FairGeoMedium* ShipMedium = media->getMedium(name);

    if (!ShipMedium) {
        Fatal("InitMedium", "Material %s not defined in media file.", name);
        return -1111;
    }
    TGeoMedium* medium = gGeoManager->GetMedium(name);
    if (medium != nullptr)
        return ShipMedium->getMediumIndex();
    return geoBuild->createMedium(ShipMedium);
}

void MTCDetector::SetMTCParameters(Double_t w,
                                   Double_t h,
                                   Double_t iron,
                                   Double_t sciFi,
                                   Double_t scint,
                                   Int_t layers,
                                   Double_t z,
                                   Double_t field)
{
    fWidth = w;
    fHeight = h;
    fIronThick = iron;
    fSciFiThick = sciFi;
    fScintThick = scint;
    fLayers = layers;
    fZCenter = z;
    fFieldY = field;
}


// Updated SciFi module builder with fiber placements
TGeoVolume* MTCDetector::CreateSegmentedLayer(const char* name,
                                              Double_t width,
                                              Double_t height,
                                              Double_t thickness,
                                              Double_t cellSizeX,
                                              Double_t cellSizeY,
                                              TGeoMedium* material,
                                              Int_t color,
                                              Double_t transparency,
                                              Int_t LayerId)
{
    auto motherVol = new TGeoVolumeAssembly(Form("%s_mother", name));
    motherVol->SetLineColor(color);
    motherVol->SetTransparency(transparency);
    auto scint_mat = new TGeoVolumeAssembly(Form("%s_scint_mat", name));
    motherVol->AddNode(scint_mat, 1, new TGeoTranslation(0, 0, 0));
    auto cell = new TGeoBBox(Form("%s_cell", name), cellSizeX / 2, cellSizeY / 2, thickness / 2);
    auto cellVol = new TGeoVolume(Form("%s_cell", name), cell, material);
    cellVol->SetLineColor(color);
    cellVol->SetTransparency(transparency);
    AddSensitiveVolume(cellVol);
    Int_t nX = Int_t(width / cellSizeX);
    Int_t nY = Int_t(height / cellSizeY);

    for (Int_t i = 0; i < nX; i++) {
        for (Int_t j = 0; j < nY; j++) {
            Double_t x = -width / 2 + cellSizeX * (i + 0.5);
            Double_t y = -height / 2 + cellSizeY * (j + 0.5);
            scint_mat->AddNode(cellVol, 1e8 + 1e6 + 2e5 + 0e4 + i * nY + j, new TGeoTranslation(x, y, 0));
        }
    }
    return motherVol;
}

// TGeoVolume* MTCDetector::DefineMat(const char* name, Int_t mat_type)
// {

//     Double_t fSciFiBendingAngle = mat_type == 1 ? 5.0 : -5.0; // degrees

//     Double_t radAngle       = fSciFiBendingAngle * TMath::DegToRad();
//     Double_t fSciFiActiveX  = width - width * tan(radAngle);
//     Double_t fSciFiActiveY  = height;
//     Double_t fiberLength    = fSciFiActiveY * cos(radAngle);
//     Double_t layerThick     = fiberMatThick / numFiberLayers;
//     Double_t fFiberRadius   = 0.01125;
//     Double_t fFiberPitch    = 0.025;
//     Int_t   fNumFibers      = static_cast<Int_t>(fSciFiActiveX / fFiberPitch);

//     // --- Lower Epoxy matrix (replaces SciFiMat layer) ---
//     TGeoBBox* epoxyMatBox = new TGeoBBox(Form("%s_Mat", name),
//                                         width/2, height/2, fiberMatThick/2);
//     TGeoVolume* ScifiMatVol = new TGeoVolume(Form("%s_Mat", name),
//                                             epoxyMatBoxU,
//                                             gGeoManager->GetMedium("Epoxy"));
//     ScifiMatVol->SetLineColor(kYellow-2);
//     ScifiMatVol->SetTransparency(30);
//     ScifiMatVol->SetVisibility(kFALSE);
//     ScifiMatVol->SetVisDaughters(kFALSE);
//     // modMotherVol->AddNode(ScifiMatVolU, 1,
//     //                     new TGeoTranslation(0, 0, zFiberMat1));

//     ScifiMatVolV->SetLineColor(kYellow-2);
//     ScifiMatVolV->SetTransparency(30);
//     ScifiMatVolV->SetVisibility(kFALSE);
//     ScifiMatVolV->SetVisDaughters(kFALSE);
//     modMotherVol->AddNode(ScifiMatVolV, 1,
//                         new TGeoTranslation(0, 0, zFiberMat2));

//     // --- Upper Internal Iron ---
//     TGeoBBox* upperIronBox = new TGeoBBox(Form("%s_upperIron", name),
//                                         width/2, height/2, upperIronThick/2);
//     TGeoVolume* upperIronVol = new TGeoVolume(Form("%s_upperIron", name),
//                                             upperIronBox,
//                                             gGeoManager->GetMedium("iron"));
//     upperIronVol->SetLineColor(kGray+1);
//     upperIronVol->SetTransparency(20);
//     modMotherVol->AddNode(upperIronVol, 1,
//                         new TGeoTranslation(0, 0, zUpperIronInt));


//     // --- Define the SciFi fiber volume ---
//     TGeoTube* fiberTube = new TGeoTube("FiberTube", 0, fFiberRadius, fiberLength/2);
//     TGeoVolume* fiberVol = new TGeoVolume("FiberVol", fiberTube,
//                                         gGeoManager->GetMedium("SciFiMat"));
//     AddSensitiveVolume(fiberVol);
//     fiberVol->SetLineColor(kMagenta);
//     fiberVol->SetTransparency(15);
//     fiberVol->SetVisibility(kFALSE);

//     // --- Rotations for U/V fibers ---
//     TGeoRotation* rotU = new TGeoRotation();
//     rotU->RotateY( fSciFiBendingAngle);
//     rotU->RotateX( 90.);
//     TGeoRotation* rotV = new TGeoRotation();
//     rotV->RotateY(-fSciFiBendingAngle);
//     rotV->RotateX( 90.);

//     // Helper to place fibers in a given mother
//     auto PlaceFibers = [&](TGeoVolume* motherVol, Int_t uvFlag) {
//     for (int layer = 0; layer < numFiberLayers; ++layer) {
//         Double_t z0 = -fiberMatThick/2 + (layer + 0.5)*(layerThick);
//         for (int j = 0; j < fNumFibers; ++j) {
//         Double_t x0 = -fSciFiActiveX/2 + (j + 0.5)*fFiberPitch;
//         TGeoCombiTrans* ct = new TGeoCombiTrans("", x0, 0, z0,
//                                                 (uvFlag==0 ? rotU : rotV));
//         Int_t copyNo = 1000000000 + LayerId*10000000 + uvFlag*100000
//                         + layer*10000 + j;
//         motherVol->AddNode(fiberVol, copyNo, ct);
//         // cout << Form("Place fiber %d U/V=%d layer=%d idx=%d",
//         //             copyNo, uvFlag, layer, j) << endl;
//         }
//     }
//     };

//     // --- Place U-fibers inside lower Epoxy ---
//     PlaceFibers(ScifiMatVolU, /*uv=*/0);

//     // --- Place V-fibers inside upper Epoxy ---
//     PlaceFibers(ScifiMatVolV, /*uv=*/1);
// }


TGeoVolume* MTCDetector::CreateSciFiModule(const char* name,
                                           Double_t width,
                                           Double_t height,
                                           Double_t thickness,
                                           Int_t LayerId)
{
    // Define sublayer thicknesses (in cm)
    // These values mimic the GEANT4 setup:
    Double_t lowerIronThick = 0.3;    // 3 mm
    Double_t fiberMatThick = 0.135;   // 1.35 mm (each fiber mat)
    Double_t airGap = 0.1;            // 1 mm
    Double_t upperIronThick = 0.3;    // 3 mm
    Double_t zLowerIronInt = -3.5 / 10;
    Double_t zFiberMat1 = -1.325 / 10;
    Double_t zAirGap = -0.15 / 10;
    Double_t zFiberMat2 = 1.025 / 10;
    Double_t zUpperIronInt = 3.2 / 10;
    // Total module thickness = 0.3 + 0.135 + 0.1 + 0.135 + 0.3 ≈ 1.0 cm

    // // Create the mother volume for the SciFi module
    // TGeoBBox* modMother = new TGeoBBox(Form("%s_mother", name), width/2, height/2, thickness/2);
    // TGeoVolume* modMotherVol = new TGeoVolume(Form("%s_mother", name), modMother, gGeoManager->GetMedium("SciFiMat"));
    // modMotherVol->SetLineColor(kGreen+2);
    // modMotherVol->SetTransparency(40);

    // // --- Lower Internal Iron ---
    // TGeoBBox* lowerIronBox = new TGeoBBox(Form("%s_lowerIron", name), width/2, height/2, lowerIronThick/2);
    // TGeoVolume* lowerIronVol = new TGeoVolume(Form("%s_lowerIron", name), lowerIronBox, gGeoManager->GetMedium("iron"));
    // lowerIronVol->SetLineColor(kGray+1);
    // lowerIronVol->SetTransparency(20);
    // modMotherVol->AddNode(lowerIronVol, 1, new TGeoTranslation(0, 0, zLowerIronInt));

    // // --- Fiber Mat U (Lower SciFi Mat) ---
    // TGeoBBox* fiberMatBoxU = new TGeoBBox(Form("%s_fiberMat_U", name), width/2, height/2, fiberMatThick/2);
    // TGeoVolume* fiberMatVolU = new TGeoVolume(Form("%s_fiberMat_U", name), fiberMatBoxU, gGeoManager->GetMedium("SciFiMat"));
    // // fiberMatVolU->SetLineColor(kYellow);
    // // fiberMatVolU->SetTransparency(30);
    // modMotherVol->AddNode(fiberMatVolU, 1, new TGeoTranslation(0, 0, zFiberMat1));

    // // --- Fiber Mat V (Upper SciFi Mat) ---
    // TGeoBBox* fiberMatBoxV = new TGeoBBox(Form("%s_fiberMat_V", name), width/2, height/2, fiberMatThick/2);
    // TGeoVolume* fiberMatVolV = new TGeoVolume(Form("%s_fiberMat_V", name), fiberMatBoxV, gGeoManager->GetMedium("SciFiMat"));
    // // fiberMatVolV->SetLineColor(kYellow);
    // // fiberMatVolV->SetTransparency(30);
    // modMotherVol->AddNode(fiberMatVolV, 1, new TGeoTranslation(0, 0, zFiberMat2));

    // // --- Upper Internal Iron ---
    // TGeoBBox* upperIronBox = new TGeoBBox(Form("%s_upperIron", name), width/2, height/2, upperIronThick/2);
    // TGeoVolume* upperIronVol = new TGeoVolume(Form("%s_upperIron", name), upperIronBox, gGeoManager->GetMedium("iron"));
    // upperIronVol->SetLineColor(kGray+1);
    // upperIronVol->SetTransparency(20);
    // modMotherVol->AddNode(upperIronVol, 1, new TGeoTranslation(0, 0, zUpperIronInt));

    // // -----------------------------
    // // Now, build the fiber arrays inside each fiber mat.
    // // Create a daughter "mother" volume in each fiber mat to hold the fibers.
    // TGeoBBox* sciFiLayerMotherUBox = new TGeoBBox(Form("%s_SciFiLayerMother_U", name), width/2, height/2, fiberMatThick/2);
    // TGeoVolume* sciFiLayerMotherUVol = new TGeoVolume(Form("%s_SciFiLayerMother_U", name), sciFiLayerMotherUBox, gGeoManager->GetMedium("SciFiMat"));
    // // sciFiLayerMotherUVol->SetLineColor(kYellow);
    // // sciFiLayerMotherUVol->SetTransparency(100);
    // // Make the mother volume invisible so that only the fibers show up
    // // sciFiLayerMotherUVol->SetVisibility(false);
    // sciFiLayerMotherUVol->SetVisDaughters(kFALSE);
    // fiberMatVolU->AddNode(sciFiLayerMotherUVol, 1, new TGeoTranslation(0, 0, 0));

    // TGeoBBox* sciFiLayerMotherVBox = new TGeoBBox(Form("%s_SciFiLayerMother_V", name), width/2, height/2, fiberMatThick/2);
    // TGeoVolume* sciFiLayerMotherVVol = new TGeoVolume(Form("%s_SciFiLayerMother_V", name), sciFiLayerMotherVBox, gGeoManager->GetMedium("SciFiMat"));
    // // sciFiLayerMotherVVol->SetLineColor(kYellow);
    // // sciFiLayerMotherVVol->SetTransparency(100);
    // // Also hide this mother volume
    // // sciFiLayerMotherVVol->SetVisibility(false);
    // sciFiLayerMotherVVol->SetVisDaughters(kFALSE);
    // fiberMatVolV->AddNode(sciFiLayerMotherVVol, 1, new TGeoTranslation(0, 0, 0));


    // // --- Define fiber parameters (in cm) ---
    Double_t fSciFiBendingAngle = 5.0; // degrees
    // Double_t radAngle = fSciFiBendingAngle * TMath::DegToRad();
    // // Assume that 80% of the module width is active for fibers.
    // Double_t fSciFiActiveAreaX = width - width  *  tan(radAngle);
    // // For the fiber length, assume the full height of the fiber mat is active.
    // Double_t fSciFiActiveAreaY = height;
    // Double_t fiberLength = fSciFiActiveAreaY * cos(radAngle);
    Int_t numFiberLayers = 6;
    // Double_t layerThickness = fiberMatThick / numFiberLayers; // thickness per fiber layer
    // Double_t fFiberRadius = 0.01125; // 0.1125 mm in cm
    // Double_t scintcore_rmax = 0.011;
    // Double_t clad1_rmin = scintcore_rmax;
    // Double_t clad1_rmax = 0.01175;
    // Double_t clad2_rmin = clad1_rmax;
    // Double_t clad2_rmax = 0.0125;

    // Double_t fFiberPitch  = 0.025*4;    // 0.25 mm in cm
    // Double_t antioverlap = 0.0;
    // Int_t fNumFibers = static_cast<Int_t>(fSciFiActiveAreaX / fFiberPitch);

    // // --- Create the fiber volume (modeled as a tube) ---
    // TGeoTube* fiberTube = new TGeoTube("FiberTube", 0, fFiberRadius, fiberLength/2);
    // TGeoVolume* fiberVol = new TGeoVolume("FiberVol", fiberTube, gGeoManager->GetMedium("SciFiMat"));
    // AddSensitiveVolume(fiberVol);
    // // AddSensitiveVolume(modMotherVol);
    // fiberVol->SetLineColor(kMagenta);
    // fiberVol->SetTransparency(15);
    // // Ensure fibers are visible
    // fiberVol->SetVisibility(true);

    // // --- Define rotations for fibers ---
    // // For the U fibers: rotate X by 90° then Y by +5°
    // TGeoRotation* rotFiberU = new TGeoRotation();
    // // 
    // rotFiberU->RotateY(fSciFiBendingAngle);
    // rotFiberU->RotateX(90.);
    // // For the V fibers: rotate X by 90° then Y by -5°
    // TGeoRotation* rotFiberV = new TGeoRotation();
    // // 
    // rotFiberV->RotateY(-fSciFiBendingAngle);
    // rotFiberV->RotateX(90.);
    // // --- Place fibers in the U fiber mat ---
    // for (int layer = 0; layer < numFiberLayers; layer++) {
    //     Double_t zCenter = -fiberMatThick/2 + (layer + 0.5) * (layerThickness + antioverlap);
    //     for (int j = 0; j < fNumFibers; j++) {
    //     Double_t xPos = -fSciFiActiveAreaX/2 + (j + 0.5) * fFiberPitch;
    //     // Create a combined translation+rotation
    //     TGeoCombiTrans* ctU = new TGeoCombiTrans("", xPos, 0, zCenter, rotFiberU);
    //     sciFiLayerMotherUVol->AddNode(fiberVol, 1000000000 + LayerId * 10000000 + 0*100000 + layer * 10000 + j, ctU);
    //     cout << "Define Scifi Fibre: " << 1000000000 + LayerId * 10000000 + 0*100000 + layer * 10000 + j << "  " << LayerId << "  " << 0 << "   " << layer << "  " << j << endl;
    //     }
    // }

    // // --- Place fibers in the V fiber mat ---
    // for (int layer = 0; layer < numFiberLayers; layer++) {
    //     Double_t zCenter = -fiberMatThick/2 + (layer + 0.5) * (layerThickness + antioverlap);
    //     for (int j = 0; j < fNumFibers; j++) {
    //     Double_t xPos = -fSciFiActiveAreaX/2 + (j + 0.5) * fFiberPitch;
    //     TGeoCombiTrans* ctV = new TGeoCombiTrans("", xPos, 0, zCenter, rotFiberV);
    //     sciFiLayerMotherVVol->AddNode(fiberVol, 1000000000 + LayerId * 10000000 + 1*100000 + layer * 10000 + j, ctV);
    //     cout << "Define Scifi Fibre: " << 1000000000 + LayerId * 10000000 + 0*100000 + layer * 10000 + j << "  " << LayerId << "  " << 1 << "   " << layer << "  " << j << endl;
    //     }
    // }

    // --- Module mother in Air ---
    TGeoVolumeAssembly* modMotherVol = new TGeoVolumeAssembly(Form("%s_mother", name));
    modMotherVol->SetLineColor(kGreen+2);
    modMotherVol->SetTransparency(40);

    // --- Lower Internal Iron ---
    TGeoBBox* lowerIronBox = new TGeoBBox(Form("%s_lowerIron", name),
                                        width/2, height/2, lowerIronThick/2);
    TGeoVolume* lowerIronVol = new TGeoVolume(Form("%s_lowerIron", name),
                                            lowerIronBox,
                                            gGeoManager->GetMedium("iron"));
    lowerIronVol->SetLineColor(kGray+1);
    lowerIronVol->SetTransparency(20);
    modMotherVol->AddNode(lowerIronVol, 1,
                        new TGeoTranslation(0, 0, zLowerIronInt));

    // --- Lower Epoxy matrix (replaces SciFiMat layer) ---
    TGeoBBox* epoxyMatBoxU = new TGeoBBox(Form("%s_epoxyMat_U", name),
                                        width/2, height/2, fiberMatThick/2);
    TGeoVolume* ScifiMatVolU = new TGeoVolume(Form("%s_epoxyMat_U", name),
                                            epoxyMatBoxU,
                                            gGeoManager->GetMedium("Epoxy"));
    ScifiMatVolU->SetLineColor(kYellow-2);
    ScifiMatVolU->SetTransparency(30);
    ScifiMatVolU->SetVisibility(kFALSE);
    ScifiMatVolU->SetVisDaughters(kFALSE);
    modMotherVol->AddNode(ScifiMatVolU, 1,
                        new TGeoTranslation(0, 0, zFiberMat1));

    // --- Upper Epoxy matrix (replaces SciFiMat layer) ---
    TGeoBBox* epoxyMatBoxV = new TGeoBBox(Form("%s_epoxyMat_V", name),
                                        width/2, height/2, fiberMatThick/2);
    TGeoVolume* ScifiMatVolV = new TGeoVolume(Form("%s_epoxyMat_V", name),
                                            epoxyMatBoxV,
                                            gGeoManager->GetMedium("Epoxy"));
    ScifiMatVolV->SetLineColor(kYellow-2);
    ScifiMatVolV->SetTransparency(30);
    ScifiMatVolV->SetVisibility(kFALSE);
    ScifiMatVolV->SetVisDaughters(kFALSE);
    modMotherVol->AddNode(ScifiMatVolV, 1,
                        new TGeoTranslation(0, 0, zFiberMat2));

    // --- Upper Internal Iron ---
    TGeoBBox* upperIronBox = new TGeoBBox(Form("%s_upperIron", name),
                                        width/2, height/2, upperIronThick/2);
    TGeoVolume* upperIronVol = new TGeoVolume(Form("%s_upperIron", name),
                                            upperIronBox,
                                            gGeoManager->GetMedium("iron"));
    upperIronVol->SetLineColor(kGray+1);
    upperIronVol->SetTransparency(20);
    modMotherVol->AddNode(upperIronVol, 1,
                        new TGeoTranslation(0, 0, zUpperIronInt));

    // -----------------------------
    // Now build the fibers inside each Epoxy block:

    // Common fiber parameters (cm)
    Double_t radAngle       = fSciFiBendingAngle * TMath::DegToRad();
    Double_t fSciFiActiveX  = width - width * tan(radAngle);
    Double_t fSciFiActiveY  = height;
    Double_t fiberLength    = fSciFiActiveY * cos(radAngle);
    Double_t layerThick     = fiberMatThick / numFiberLayers;
    // Double_t scintcore_rmax = 0.011;
    // Double_t clad1_rmin = scintcore_rmax;
    // Double_t clad1_rmax = 0.01175;
    // Double_t clad2_rmin = clad1_rmax;
    // Double_t clad2_rmax = 0.0125;
    Double_t fFiberRadius   = 0.01124;
    Double_t fFiberPitch    = 0.025;
    Int_t   fNumFibers      = static_cast<Int_t>(fSciFiActiveX / fFiberPitch);

    // --- Define the SciFi fiber volume ---
    TGeoTube* fiberTube = new TGeoTube("FiberTube", 0, fFiberRadius, fiberLength/2);
    TGeoVolume* fiberVol = new TGeoVolume("FiberVol", fiberTube,
                                        gGeoManager->GetMedium("SciFiMat"));
    AddSensitiveVolume(fiberVol);
    fiberVol->SetLineColor(kMagenta);
    fiberVol->SetTransparency(15);
    fiberVol->SetVisibility(kFALSE);

    // --- Rotations for U/V fibers ---
    TGeoRotation* rotU = new TGeoRotation();
    rotU->RotateY( fSciFiBendingAngle);
    rotU->RotateX( 90.);
    TGeoRotation* rotV = new TGeoRotation();
    rotV->RotateY(-fSciFiBendingAngle);
    rotV->RotateX( 90.);

    // --- Place U-fibers inside lower Epoxy ---
    for (int layer = 0; layer < numFiberLayers; ++layer) {
        Double_t z0 = -fiberMatThick / 2 + (layer + 0.5) * (layerThick);
        for (int j = 0; j < fNumFibers; ++j) {
            Double_t x0 = -fSciFiActiveX / 2 + (j + 0.5) * fFiberPitch;
            TGeoCombiTrans* ct = new TGeoCombiTrans("", x0, 0, z0, rotU);
            Int_t copyNo = 100000000 + 1000000 + 0 * 100000 + layer * 10000 + j;
            ScifiMatVolU->AddNode(fiberVol, copyNo, ct);
            // cout << Form("Place fiber %d U layer=%d idx=%d", copyNo, layer, j) << endl;
        }
    }

    // --- Place V-fibers inside upper Epoxy ---
    for (int layer = 0; layer < numFiberLayers; ++layer) {
        Double_t z0 = -fiberMatThick / 2 + (layer + 0.5) * (layerThick);
        for (int j = 0; j < fNumFibers; ++j) {
            Double_t x0 = -fSciFiActiveX / 2 + (j + 0.5) * fFiberPitch;
            TGeoCombiTrans* ct = new TGeoCombiTrans("", x0, 0, z0, rotV);
            Int_t copyNo = 100000000 + 1000000 + 1 * 100000 + layer * 10000 + j;
            ScifiMatVolV->AddNode(fiberVol, copyNo, ct);
            // cout << Form("Place fiber %d V layer=%d idx=%d", copyNo, layer, j) << endl;
        }
    }
    return modMotherVol;
}


void Scifi::SiPMOverlap()
{
    if (gGeoManager->FindVolumeFast("SiPMmapVol")){return;}
	Double_t fLengthScifiMat = conf_floats["Scifi/scifimat_length"];
	Double_t fWidthChannel = conf_floats["Scifi/channel_width"];
	Double_t fZEpoxyMat          = conf_floats["Scifi/epoxymat_z"];
	Int_t fNSiPMChan = conf_ints["Scifi/nsipm_channels"];
	Int_t fNSiPMs  = conf_ints["Scifi/nsipm_mat"];
	Int_t fNMats   = conf_ints["Scifi/nmats"];
	Double_t fEdge = conf_floats["Scifi/sipm_edge"];
	Double_t fCharr = conf_floats["Scifi/charr_width"];
	Double_t fCharrGap = conf_floats["Scifi/charr_gap"];
	Double_t fBigGap = conf_floats["Scifi/sipm_diegap"];
	Double_t firstChannelX = conf_floats["Scifi/firstChannelX"];

    //Contains all plane SiPMs, defined for horizontal fiber plane
    //To obtain SiPM map for vertical fiber plane rotate by 90 degrees around Z
    TGeoVolumeAssembly *SiPMmapVol = new TGeoVolumeAssembly("SiPMmapVol");

    TGeoVolume*ChannelVol = gGeoManager->MakeBox("ChannelVol", 0, fLengthScifiMat/2, fWidthChannel/2, fZEpoxyMat/2);

    //DetID for each channel:
    //first digit: mat number (0-2)
    //second digit: SiPM number (0-3)
    //last three digits: channel number (0-127)

    Double_t SiPMArray_fullwidth = fEdge+fCharr+fCharrGap+fCharr+fEdge;
    TGeoVolumeAssembly *SiPMArrayVol;
    int N = fNMats == 1 ? 1 : 0;
    Double_t pos = -fEdge+firstChannelX + N*fLengthScifiMat;
    for (int imat = 0; imat < fNMats; imat++){
      for (int isipms = 0; isipms < fNSiPMs; isipms++){
        pos+= fEdge;
        for (int ichannel = 0; ichannel < fNSiPMChan; ichannel++){
            SiPMmapVol->AddNode(ChannelVol, imat*10000+isipms *1000 + ichannel, new TGeoTranslation(0, pos, 0));
            pos += fWidthChannel;
            if (ichannel==(fNSiPMChan/2-1)){pos += fCharrGap;}
        }
        pos+=fEdge+fBigGap;
    }
   }
}

void Scifi::GetPosition(Int_t fDetectorID, TVector3& A, TVector3& B) 
{
//	TGeoVolumeAssembly *SiPMmapVol = gGeoManager->FindVolumeFast("SiPMmapVol");
//	if(!SiPMmapVol ){SiPMmapVol=SiPMOverlap();}

/* STMRFFF
 First digit S: 			station # within the sub-detector
 Second digit T: 		type of the plane: 0-horizontal fiber plane, 1-vertical fiber plane
 Third digit M: 			determines the mat number
 Fourth digit R: 		row number (in Z direction)
 Last three digits F: 	fiber number
*/

	Int_t station_number = int(fDetectorID/1e6);
	Int_t mat_number = int(fDetectorID/1e4)%int(fDetectorID/1e5);

	Int_t local_fibre_id = fDetectorID - (station_number-1)*1e6 - (mat_number-1)*1e4;
	TString sLocalID;
	sLocalID.Form("%i", local_fibre_id);

	TString sID;
	sID.Form("%i",fDetectorID);
	TString path = "/cave_1/Detector_0/volTarget_1/ScifiVolume"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000/";
	if (sID(1,1)=="0"){
		path+="ScifiHorPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000/";
		path+="HorMatVolume_"+TString(sID(0,3))+"0000/";
	}else{
		path+="ScifiVertPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000/";
		path+="VertMatVolume_"+TString(sID(0,3))+"0000/";
	}
	path+="FiberVolume_"+sLocalID;
	TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
	nav->cd(path);
	LOG(DEBUG) <<path<<" "<<fDetectorID;
	TGeoNode* W = nav->GetCurrentNode();
	TGeoBBox* S = dynamic_cast<TGeoBBox*>(W->GetVolume()->GetShape());

	Double_t top[3] = {0,0,S->GetDZ()};
	Double_t bot[3] = {0,0,-(S->GetDZ())};
	Double_t Gtop[3],Gbot[3];
	nav->LocalToMaster(top, Gtop);   nav->LocalToMaster(bot, Gbot);
	A.SetXYZ(Gtop[0],Gtop[1],Gtop[2]);
	B.SetXYZ(Gbot[0],Gbot[1],Gbot[2]);

}
TVector3 Scifi::GetLocalPos(Int_t id, TVector3* glob){
	TString sID;
	sID.Form("%i",id);
	TString path = "/cave_1/Detector_0/volTarget_1/ScifiVolume"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000/";
	if (sID(1,1)=="0"){
		path+="ScifiHorPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000";
	}else{
		path+="ScifiVertPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000";
	}
	TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
	nav->cd(path);
	Double_t aglob[3];
	Double_t aloc[3];
	glob->GetXYZ(aglob);
	nav->MasterToLocal(aglob,aloc);
	return TVector3(aloc[0],aloc[1],aloc[2]);
}

void Scifi::GetSiPMPosition(Int_t SiPMChan, TVector3& A, TVector3& B)
{
/* STMRFFF
 First digit S: 		station # within the sub-detector
 Second digit T: 		type of the plane: 0-horizontal fiber plane, 1-vertical fiber plane
 Third digit M: 		determines the mat number 0-2
 Fourth digit S: 		SiPM number  0-3
 Last three digits F: 	local SiPM channel number in one mat  0-127
*/
	Int_t locNumber            = SiPMChan%100000;
	Int_t globNumber         = int(SiPMChan/100000)*100000;
	Float_t locPosition        = SiPMPos[locNumber]; // local position in plane of reference plane.
	Double_t fFiberLength  = conf_floats["Scifi/fiber_length"];
	Int_t fNMats   = conf_ints["Scifi/nmats"];

	TString tag = "";

	// in case of old data with FairEventHeader, user will be responsible to use the correct geofile.
	if (eventHeader){
		Int_t fRunNumber = eventHeader->GetRunId();
		if (fRunNumber != last_run_pos){
		  last_run_pos = fRunNumber;

		  if (fRunNumber<1) {
		  LOG(ERROR) << "Scifi::GetSiPMPosition: non valid run number "<<fRunNumber;
		  return;
		  }

		  if (covered_runs_position_alignment.size()!=0){
		      tag = "t_"+std::to_string(covered_runs_position_alignment[covered_runs_position_alignment.size()-1]);
		      for (int i=1; i<covered_runs_position_alignment.size(); i++){
		           if (fRunNumber>=covered_runs_position_alignment[i-1] && fRunNumber<covered_runs_position_alignment[i]){
		               tag = "t_"+std::to_string(covered_runs_position_alignment[i-1]);
		           }
		      }
		  }
		  else{
		      // allow reading older geo files with letter tags i.e. A, B, C
		      tag = "E";
		      if (fRunNumber<4575) {tag = "A";}
		      else if (fRunNumber<4855) {tag = "B";}
		      else if (fRunNumber<5172) {tag = "C";}
		      else if (fRunNumber<5431) {tag = "D";}
		  }
		  // 2023 testbeam data doesn't have a custom tag
		  if (fRunNumber>=1e5) {tag = "";}
		  last_position_alignment_tag = tag;
		}
	}
	TString sID;
	sID.Form("%i",SiPMChan);
	Int_t digits = fNMats==1 ? 2 : 1;
	locPosition += conf_floats["Scifi/LocM"+TString(sID(0,3))+last_position_alignment_tag];
	Float_t rotPhi = conf_floats["Scifi/RotPhiS"+TString(sID(0,digits))+last_position_alignment_tag];
	Float_t rotPsi = conf_floats["Scifi/RotPsiS"+TString(sID(0,digits))+last_position_alignment_tag];
	Float_t rotTheta = conf_floats["Scifi/RotThetaS"+TString(sID(0,digits))+last_position_alignment_tag];

	Double_t loc[3] = {0,0,0};
	TString path = "/cave_1/Detector_0/volTarget_1/ScifiVolume"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000/";
	TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
	Double_t glob[3] = {0,0,0};

	if (sID(1,1)=="0"){
		path+="ScifiHorPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000";
		loc[0] = -fFiberLength/2 - (rotPhi + rotPsi)*locPosition ;
		loc[1] = locPosition - fFiberLength/2 * (rotPhi + rotPsi) ;
		loc[2] = rotTheta*locPosition;
		nav->cd(path);
		nav->LocalToMaster(loc, glob);
		A.SetXYZ( glob[0], glob[1],glob[2] );
		loc[0] = fFiberLength/2 - (rotPhi + rotPsi)*locPosition ;
		loc[1] = locPosition + fFiberLength/2 * (rotPhi + rotPsi) ;
		loc[2] = - rotTheta*locPosition;
		nav->LocalToMaster(loc, glob);
		B.SetXYZ( glob[0], glob[1],glob[2] );
	}else{
		path+="ScifiVertPlaneVol"+TString(sID(0,1))+"_"+TString(sID(0,1))+"000000";
		loc[0] = locPosition + fFiberLength/2*(rotPhi + rotPsi);
		loc[1] = -fFiberLength/2 + locPosition*(rotPhi + rotPsi);
		loc[2] = -fFiberLength/2*rotTheta;
		nav->cd(path);
		nav->LocalToMaster(loc, glob);
		A.SetXYZ( glob[0], glob[1],glob[2] );
		loc[0] = locPosition - fFiberLength/2*(rotPhi + rotPsi);
		loc[1] = fFiberLength/2 + locPosition*(rotPhi + rotPsi);
		loc[2] = -fFiberLength/2*rotTheta;
		nav->LocalToMaster(loc, glob);
		B.SetXYZ( glob[0], glob[1],glob[2] );
	}
}

Double_t Scifi::ycross(Double_t a,Double_t R,Double_t x)
{
	Double_t y = -1;
	Double_t A = R*R - (x-a)*(x-a);
	if ( !(A<0) ){y = TMath::Sqrt(A);}
	return y;
}
Double_t Scifi::integralSqrt(Double_t ynorm)
{
	Double_t y = 1./2.*(ynorm*TMath::Sqrt(1-ynorm*ynorm)+TMath::ASin(ynorm));
	return y;
}
Double_t Scifi::fraction(Double_t R,Double_t x,Double_t y)
{
	Double_t F= 2*R*R*(integralSqrt(y/R) );
	F-=(2*x*y);
	Double_t result = F/(R*R*TMath::Pi());
       return  result;
}
Double_t Scifi::area(Double_t a,Double_t R,Double_t xL,Double_t xR)
{
	Double_t fracL = -1;
	Double_t fracR = -1;
	if (xL<=a-R && xR>=a+R) {return 1;}
	Double_t leftC    = ycross(a,R,xL);
	Double_t rightC = ycross(a,R,xR);
	if (leftC<0 && rightC<0) {return -1;}
	if ( !(rightC<0) ){  fracR = fraction(R,abs(xR-a),rightC);}
	if ( !(leftC<0) )   {   fracL = fraction(R,abs(xL-a),leftC);}
	Double_t theAnswer = 0;
	if ( !(leftC<0) ) {
		if(xL<a){theAnswer += 1-fracL;}
		else{      theAnswer += fracL;}
		if ( !(rightC<0) ) {theAnswer -=1;}
	}
	if ( !(rightC<0) ){
		if(xR>a){ theAnswer += 1-fracR;}
		else{      theAnswer +=  fracR;}
	}
	return theAnswer;
}

void Scifi::SiPMmapping(){
	Float_t fibresRadius = -1;
	Float_t dSiPM = -1;
	TGeoNode* vol;
	TGeoNode* fibre;
	SiPMOverlap();           // 12 SiPMs per mat, made for horizontal mats, fibres staggered along y-axis.
	auto sipm    = gGeoManager->FindVolumeFast("SiPMmapVol");
	TObjArray* Nodes = sipm->GetNodes();
	auto plane  = gGeoManager->FindVolumeFast("ScifiHorPlaneVol1");
	for (int imat = 0; imat < plane->GetNodes()->GetEntriesFast(); imat++){
		auto mat =  static_cast<TGeoNode*>(plane->GetNodes()->At(imat));
		Float_t t1 = mat->GetMatrix()->GetTranslation()[1];
		auto vmat = mat->GetVolume();
		for (int ifibre = 0; ifibre < vmat->GetNodes()->GetEntriesFast(); ifibre++){
			fibre = static_cast<TGeoNode*>(vmat->GetNodes()->At(ifibre));
			if  (fibresRadius<0){
				auto tmp = fibre->GetVolume()->GetShape();
				auto S = dynamic_cast<TGeoBBox*>(tmp);
				fibresRadius = S->GetDX();
			}
			Float_t t2 = fibre->GetMatrix()->GetTranslation()[1];
			Int_t fID = fibre->GetNumber()%100000 + imat*1e4;     // local fibre number, global fibre number = SO+fID
			Float_t a = t1+t2;

	//  check for overlap with any of the SiPM channels in the same mat
			for(Int_t nChan = 0; nChan< Nodes->GetEntriesFast();nChan++){        // 12 SiPMs total and 4 SiPMs per mat times 128 channels
				vol = static_cast<TGeoNode*>(Nodes->At(nChan));
				Int_t N = vol->GetNumber()%100000;
				if (imat!=int(N/10000)){continue;}
				Float_t xcentre = vol->GetMatrix()->GetTranslation()[1];
				if  (dSiPM<0){
					TGeoBBox* B = dynamic_cast<TGeoBBox*>(vol->GetVolume()->GetShape());
					dSiPM = B->GetDY();
				}
				if (TMath::Abs(xcentre-a)>4*fibresRadius){ continue;} // no need to check further
				Float_t W = area(a,fibresRadius,xcentre-dSiPM,xcentre+dSiPM);
				if (W<0){ continue;}
				std::array<float, 2> Wa;
				Wa[0] = W;
				Wa[1] = a;
				fibresSiPM[N][fID] = Wa;
			}
		}
	}
  // calculate also local SiPM positions based on fibre positions and their fraction
  // probably an overkill, maximum difference between weighted average and central position < 6 micron.
	std::map<Int_t,std::map<Int_t,std::array<float, 2>>>::iterator it;
	std::map<Int_t,std::array<float, 2>>::iterator itx;
	for (it = fibresSiPM.begin(); it != fibresSiPM.end(); it++)
	{
		Int_t N = it->first;
		Float_t m = 0;
		Float_t w = 0;
		for (itx = it->second.begin(); itx != it->second.end(); itx++)
		{
			m+=(itx->second)[0]*(itx->second)[1];
			w+=(itx->second)[0];
		}
		SiPMPos[N]=m/w;
	}
// make inverse mapping, which fibre is associated to which SiPMs
	for (it = fibresSiPM.begin(); it != fibresSiPM.end(); it++)
	{
		Int_t N = it->first;
		for (itx = it->second.begin(); itx != it->second.end(); itx++)
		{
			Int_t nfibre = itx->first;
			siPMFibres[nfibre][N]=itx->second;
		}
	}
}


void MTCDetector::ConstructGeometry()
{
    // Initialize media (using FairROOT’s interface)
    InitMedium("SciFiMat");
    TGeoMedium* SciFiMat = gGeoManager->GetMedium("SciFiMat");
    InitMedium("Epoxy");
    TGeoMedium* Epoxy = gGeoManager->GetMedium("Epoxy");
    InitMedium("air");
    TGeoMedium* air = gGeoManager->GetMedium("air");
    TGeoMedium* ironMed = gGeoManager->GetMedium("iron");
    // For the scintillator, you may use the same medium as SciFiMat or another if defined.
    TGeoMedium* scintMed = gGeoManager->GetMedium("SciFiMat");


    // Define the module spacing based on three sublayers:
    //   fIronThick (outer iron), fSciFiThick (SciFi module/fiber module), fScintThick (scintillator)
    Double_t moduleSpacing = fIronThick + fSciFiThick + fScintThick;
    Double_t totalLength = fLayers * moduleSpacing;

    // --- Create an envelope volume for the detector (green, semi-transparent) ---
    auto envBox = new TGeoBBox("MTC_env", fWidth / 2, fHeight / 2, totalLength / 2);
    auto envVol = new TGeoVolume("MTC", envBox, air);
    envVol->SetLineColor(kGreen);
    envVol->SetTransparency(50);

    // --- Outer Iron Layer (gray) ---
    auto ironBox = new TGeoBBox("MTC_iron", fWidth / 2, fHeight / 2, fIronThick / 2);
    auto ironVol = new TGeoVolume("MTC_iron", ironBox, ironMed);
    ironVol->SetLineColor(kGray + 1);
    ironVol->SetTransparency(20);
    // Enable the field in the iron volume
    if (fFieldY != 0) ironVol->SetField(new TGeoUniformMagField(0, fFieldY, 0));

    // --- Assemble the layers into the envelope ---

    // Define a layer for the SciFi module
    TGeoVolume* sciFiModuleVol = CreateSciFiModule("MTC_sciFi", fWidth, fHeight, fSciFiThick, 1);
    TGeoVolume* scintVol =
            CreateSegmentedLayer("MTC_scint", fWidth, fHeight, fScintThick, 1.0, 1.0, scintMed, kAzure + 7, 30, 1);
    TGeoVolumeAssembly* sensitiveModule = new TGeoVolumeAssembly("MTC_layer");
    sensitiveModule->AddNode(sciFiModuleVol, 1, new TGeoTranslation(0, 0, 0));
    sensitiveModule->AddNode(scintVol, 2, new TGeoTranslation(0, 0, fSciFiThick / 2 + fScintThick / 2));
    for (Int_t i = 0; i < fLayers; i++) {
        // Compute the center position (z) for the current module
        Double_t zPos = -totalLength / 2 + i * moduleSpacing;

        // Place the Outer Iron layer (shifted down by half the SciFi+scint thickness)
        envVol->AddNode(ironVol, i, new TGeoTranslation(0, 0, zPos + fIronThick / 2));
        // Create a SciFi module with the current detector id 'i'
        // TGeoVolume* sciFiModuleVol = CreateSciFiModule("MTC_sciFi", fWidth, fHeight, fSciFiThick, i);
        // Double_t scifi_layer_num = 1000000000 + i*10000000;
        // envVol->AddNode(sciFiModuleVol, scifi_layer_num, new TGeoTranslation(0, 0, zPos + fIronThick + fSciFiThick / 2));
        // TGeoVolume* scintVol =
        //     CreateSegmentedLayer("MTC_scint", fWidth, fHeight, fScintThick, 1.0, 1.0, scintMed, kAzure + 7, 30, i);
        // Place the Scintillator layer (shifted up by half the iron thickness)
        // Double_t scint_layer_num = 2000000000 + i*10000000;

        // envVol->AddNode(scintVol, i, new TGeoTranslation(0, 0, zPos + fIronThick + fSciFiThick + fScintThick / 2));
        envVol->AddNode(sensitiveModule, i, new TGeoTranslation(0, 0, zPos + fIronThick + fSciFiThick / 2));
    }

    // Finally, add the envelope to the top volume with the global z offset fZCenter
    gGeoManager->GetTopVolume()->AddNode(envVol, 1, new TGeoTranslation(0, 0, fZCenter));
}
// Standard FairDetector methods
void MTCDetector::Initialize()
{
    FairDetector::Initialize();
}

Bool_t MTCDetector::ProcessHits(FairVolume* vol)
{
    /** This method is called from the MC stepping */
    // Set parameters at entrance of volume. Reset ELoss.
    if (gMC->IsTrackEntering()) {
        fELoss = 0.;
        fTime = gMC->TrackTime() * 1.0e09;
        fLength = gMC->TrackLength();
        gMC->TrackPosition(fPos);
        gMC->TrackMomentum(fMom);
        TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
        Int_t vol_local_id = nav->GetCurrentNode()->GetNumber() % 1000000; // Local ID within the mat or scint.
        // cout << "detID_1: " << detID_1 << "  " << fELoss << endl;
		Int_t layer_id = nav->GetMother(3)->GetNumber(); // Get layer ID.
        fVolumeID = 100000000 + layer_id * 1000000 + vol_local_id; // 1e8 + layer_id * 1e6 + fibre_local_id;
        cout << "MTCDetector::ProcessHits: fVolumeID = " << fVolumeID << "  " << nav->GetMother(3)->GetNumber() << endl;
    }
    // Sum energy loss for all steps in the active volume
    fELoss += gMC->Edep();

    // Create vetoPoint when exiting active volume
    if (gMC->IsTrackExiting() || gMC->IsTrackStop() || gMC->IsTrackDisappeared()) {

        if (fELoss == 0.) {
            return kFALSE;
        }   // if you do not want hits with zero eloss

        TParticle* p = gMC->GetStack()->GetCurrentTrack();
        fTrackID = gMC->GetStack()->GetCurrentTrackNumber();
        Int_t pdgCode = p->GetPdgCode();
        Int_t detID_1;
        // TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
        gMC->CurrentVolID(detID_1);
		// Int_t vol_local_id = nav->GetNumber() % 1000000; // Local ID within the mat or scint.
        // Int_t vol_local_id = detID_1 % 1000000;
        cout << "detID_1: " << detID_1 << "  " << fELoss << endl;
		// Int_t layer_id = nav->GetMother()->GetNumber(); // Get layer ID.
        // Int_t detID = 100000000 + layer_id * 1000000 + vol_local_id; // 1e8 + layer_id * 1e6 + fibre_local_id;

        // cout << "MTCDetector::ProcessHits: detID = " << detID << "  " << detID_1 << "  " << nav->GetMother()->GetNumber() << endl;
        // 1) get current node
        // TGeoNode* curr = nav->GetCurrentNode();
        // if (curr) {
        //     std::cout << " Level(0) (current): name=\""
        //             << curr->GetName() << "\"  id="
        //             << curr->GetNumber() << "  " << detID_1 << "  " << fELoss << "\n";
        // } else {
        //     std::cout << " Level(0) (current): NULL\n";
        // }
        // Int_t depth = nav->GetLevel();            // how many steps from top
        // std::cout << "Current depth: " << depth << "\n";
        // for (int lvl = 1; lvl <= depth; ++lvl) {
        //     TGeoNode* mom = nav->GetMother(lvl);
        //     if (mom) {
        //         std::cout << " Mother(" << lvl << ") = \""
        //                 << mom->GetName() << "\"  id="
        //                 << mom->GetNumber()  << "\n";
        //     } else {
        //         std::cout << " Mother(" << lvl << ") = NULL\n";
        //     }
        // }
        TLorentzVector Pos;
        gMC->TrackPosition(Pos);
        TLorentzVector Mom;
        gMC->TrackMomentum(Mom);
        Double_t x, y, z;
        if (fVolumeID / 100000 == 3) {
            x = (fPos.X() + Pos.X()) / 2.;
            y = (fPos.Y() + Pos.Y()) / 2.;
            z = (fPos.Z() + Pos.Z()) / 2.;
            }
            else {
            x = fPos.X();
            y = fPos.Y();
            z = (fPos.Z() + Pos.Z()) / 2.;
            }

        AddHit(fTrackID,
                fVolumeID,
                TVector3(x, y, z),
                TVector3(fMom.Px(), fMom.Py(), fMom.Pz()),   // entrance momentum
                fTime,
                fLength,
                fELoss,
                pdgCode);
        ShipStack* stack = dynamic_cast<ShipStack*>(gMC->GetStack());
        stack->AddPoint(kMTC);
    }
    return kTRUE;
}

void MTCDetector::Register()
{
    TString name = "MtcDetPoint";
    TString title = "MTC";
    FairRootManager::Instance()->Register(name, title, fMTCDetectorPointCollection, kTRUE);
    LOG(DEBUG) << this->GetName() << ", Register() says: registered " << name << " collection";
}

TClonesArray* MTCDetector::GetCollection(Int_t iColl) const
{
    if (iColl == 0) {
        return fMTCDetectorPointCollection;
    } else {
        return NULL;
    }
}

void MTCDetector::Reset()
{
    fMTCDetectorPointCollection->Clear();
}

void MTCDetector::EndOfEvent()
{
    fMTCDetectorPointCollection->Clear();
}

MtcDetPoint* MTCDetector::AddHit(Int_t trackID,
                                 Int_t detID,
                                 TVector3 pos,
                                 TVector3 mom,
                                 Double_t time,
                                 Double_t length,
                                 Double_t eLoss,
                                 Int_t pdgCode)
{
    TClonesArray& clref = *fMTCDetectorPointCollection;
    Int_t size = clref.GetEntriesFast();
    return new (clref[size]) MtcDetPoint(trackID, detID, pos, mom, time, length, eLoss, pdgCode);
}
