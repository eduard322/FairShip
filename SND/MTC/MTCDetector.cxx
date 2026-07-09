// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright CERN for the benefit of the SHiP
// Collaboration

// MTC detector specific headers
#include "MTCDetector.h"

#include "MTCDetPoint.h"
#include "ShipDetectorList.h"
#include "ShipGeoUtil.h"
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
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairVolume.h"

// Additional standard headers
#include "TList.h"      // for TListIter, TList (ptr only)
#include "TObjArray.h"  // for TObjArray
#include "TString.h"    // for TString
#include "TVirtualMC.h"

namespace {
Double_t ycross(Double_t a, Double_t R, Double_t x) {
  /*
   * ycross:
   *   Compute the positive y-coordinate where the vertical line x intersects
   *   the circle of radius R centered at (a, 0). If the line does not intersect
   *   (i.e., (x-a)^2 > R^2), returns -1 as a flag.
   */
  Double_t y = -1;
  Double_t A = R * R - (x - a) * (x - a);
  if (!(A < 0)) {
    y = TMath::Sqrt(A);
  }
  return y;
}

Double_t integralSqrt(Double_t ynorm) {
  /*
   * integralSqrt:
   *   Compute the analytic integral ∫₀^{ynorm} sqrt(1 - t^2) dt
   *   = ½ [ ynorm * sqrt(1 - ynorm^2) + arcsin(ynorm) ].
   *   This is used for normalizing the circular segment area.
   */
  Double_t y =
      1. / 2. * (ynorm * TMath::Sqrt(1 - ynorm * ynorm) + TMath::ASin(ynorm));
  return y;
}

Double_t fraction(Double_t R, Double_t x, Double_t y) {
  /*
   * fraction:
   *   Compute the fraction of the circle's total area that lies on one side
   *   of a vertical cut at horizontal distance x from the circle center,
   *   up to the intersection height y = sqrt(R^2 - x^2).
   *   Formula:
   *     F = 2 R^2 ∫₀^{y/R} sqrt(1 - t^2) dt  -  2 x y
   *     result = F / (π R^2)
   */
  Double_t F = 2 * R * R * (integralSqrt(y / R));
  F -= (2 * x * y);
  Double_t result = F / (R * R * TMath::Pi());
  return result;
}

Double_t area(Double_t a, Double_t R, Double_t xL, Double_t xR) {
  /*
   * area:
   *   Compute the fraction of the full circle (radius R, center at (a,0))
   *   that lies between the vertical boundaries x = xL and x = xR.
   *   Special cases:
   *     - If [xL, xR] fully covers the circle, returns 1.
   *     - If neither boundary intersects the circle, returns -1 (no overlap).
   *   Otherwise, uses ycross() to find intersection heights, fraction()
   *   to get segment areas, and combines them to yield the net fraction.
   */
  Double_t fracL = -1;
  Double_t fracR = -1;
  if (xL <= a - R && xR >= a + R) {
    return 1;
  }

  Double_t leftC = ycross(a, R, xL);
  Double_t rightC = ycross(a, R, xR);
  if (leftC < 0 && rightC < 0) {
    return -1;
  }

  if (!(rightC < 0)) {
    fracR = fraction(R, TMath::Abs(xR - a), rightC);
  }
  if (!(leftC < 0)) {
    fracL = fraction(R, TMath::Abs(xL - a), leftC);
  }

  Double_t theAnswer = 0;
  if (!(leftC < 0)) {
    if (xL < a) {
      theAnswer += 1 - fracL;
    } else {
      theAnswer += fracL;
    }
    if (!(rightC < 0)) {
      theAnswer -= 1;
    }
  }
  if (!(rightC < 0)) {
    if (xR > a) {
      theAnswer += 1 - fracR;
    } else {
      theAnswer += fracR;
    }
  }
  return theAnswer;
}
}  // namespace

MTCDetector::MTCDetector() : Detector("MTC", kTRUE, kMTC) {}

MTCDetector::MTCDetector(const char* name, Bool_t Active, const char* /*Title*/,
                         Int_t /*DetId*/)
    : Detector(name, Active, kMTC) {}

void MTCDetector::SetMTCParameters(
    Double_t width, Double_t height, Double_t fiber_tilt_angle,
    Double_t iron_thickness, Double_t scifi_thickness,
    Int_t num_of_agg_channels, Double_t scint_cell_size,
    Double_t scint_thickness, Int_t number_of_layers, Double_t z_position,
    Double_t field_strength) {
  fWidth = width;
  fHeight = height;
  fSciFiBendingAngle = fiber_tilt_angle;
  fIronThick = iron_thickness;
  fSciFiThick = scifi_thickness;
  fChannelAggregated = num_of_agg_channels;
  fScintCellSize = scint_cell_size;
  fScintThick = scint_thickness;
  fLayers = number_of_layers;
  fZCenter = z_position;
  fFieldY = field_strength;
  fSciFiActiveX = fWidth * (1 - tan(fSciFiBendingAngle * TMath::DegToRad()));
  fSciFiActiveY = fHeight;
  //Do 3 different blocks with 40x40, 50x50 then 60x60 sizes, each nLayers/3
  fnB = 3;
  fnLayPerBlock = fLayers / fnB;

  // Per-block transverse sizes (smallest upstream): {40,50,60} for width 60.
  // NOTE: hardcoded for fnB == 3.
  fBlockWidth = {fWidth - 20, fWidth - 10, fWidth};
  fBlockHeight = {fHeight - 20, fHeight - 10, fHeight};

  // Size all per-block containers and precompute the (height-dependent) fibre
  // length so it is available whenever the detector is configured (both at
  // simulation and at digitisation time, independent of ConstructGeometry).
  fFiberLength.assign(fnB, 0.);
  fNSiPMChan.assign(fnB, 0);
  fNSiPMs.assign(fnB, 1);
  fibresSiPM_U.assign(fnB, {});
  siPMFibres_U.assign(fnB, {});
  fibresSiPM_V.assign(fnB, {});
  siPMFibres_V.assign(fnB, {});
  SiPMPos_U.assign(fnB, {});
  SiPMPos_V.assign(fnB, {});
  for (Int_t iB = 0; iB < fnB; ++iB) {
    fFiberLength[iB] =
        fBlockHeight[iB] / cos(fSciFiBendingAngle * TMath::DegToRad()) -
        2 * fFiberRadius * sin(fSciFiBendingAngle * TMath::DegToRad());
  }
}

// Updated SciFi module builder with fiber placements
void MTCDetector::CreateScintModule(const char* name,
                                    TGeoVolumeAssembly* modMotherVol,
                                    Double_t z_shift, Double_t width,
                                    Double_t height, Double_t thickness,
                                    Double_t cellSizeX, Double_t cellSizeY,
                                    TGeoMedium* material, Int_t color,
                                    Double_t transparency, Int_t LayerId) {
  modMotherVol->SetLineColor(color);
  modMotherVol->SetTransparency(transparency);
  auto scint_volume = new TGeoVolumeAssembly(Form("%s_scint", name));
  modMotherVol->AddNode(scint_volume, 0, new TGeoTranslation(0, 0, z_shift));
  auto scint_mat = new TGeoVolumeAssembly(Form("%s_scint_mat", name));
  scint_volume->AddNode(scint_mat, 0, new TGeoTranslation(0, 0, 0));
  auto cell = new TGeoBBox(Form("%s_cell", name), cellSizeX / 2, cellSizeY / 2,
                           thickness / 2);
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
      scint_mat->AddNode(cellVol, 1e8 + 1e6 + 2e5 + 0e4 + i * nY + j,
                         new TGeoTranslation(x, y, 0));
    }
  }
}

void MTCDetector::CreateSciFiModule(const char* name,
                                    TGeoVolumeAssembly* modMotherVol,
                                    Double_t width, Double_t height,
                                    Double_t thickness, Int_t iBlock) {
  // --- Lower Internal Iron ---
  TGeoBBox* lowerIronBox = new TGeoBBox(Form("%s_lowerIron", name), width / 2,
                                        height / 2, lowerIronThick / 2);
  TGeoVolume* lowerIronVol = new TGeoVolume(
      Form("%s_lowerIron", name), lowerIronBox, gGeoManager->GetMedium("iron"));
  lowerIronVol->SetLineColor(kGray + 1);
  lowerIronVol->SetTransparency(20);
  modMotherVol->AddNode(lowerIronVol, 0,
                        new TGeoTranslation(0, 0, zLowerIronInt));

  // --- Lower Epoxy matrix (replaces SciFiMat layer) ---
  TGeoVolumeAssembly* ScifiVolU =
      new TGeoVolumeAssembly(Form("%s_scifi_U", name));
  modMotherVol->AddNode(ScifiVolU, 0, new TGeoTranslation(0, 0, 0));
  TGeoBBox* epoxyMatBoxU = new TGeoBBox(Form("%s_epoxyMat", name), width / 2,
                                        height / 2, fiberMatThick / 2);
  TGeoVolume* ScifiMatVolU = new TGeoVolume(
      Form("%s_epoxyMat", name), epoxyMatBoxU, gGeoManager->GetMedium("Epoxy"));
  ScifiMatVolU->SetLineColor(kYellow - 2);
  ScifiMatVolU->SetTransparency(30);
  ScifiMatVolU->SetVisibility(kFALSE);
  ScifiMatVolU->SetVisDaughters(kFALSE);
  ScifiVolU->AddNode(ScifiMatVolU, 0, new TGeoTranslation(0, 0, zFiberMat1));

  // --- Upper Epoxy matrix (replaces SciFiMat layer) ---
  TGeoVolumeAssembly* ScifiVolV =
      new TGeoVolumeAssembly(Form("%s_scifi_V", name));
  modMotherVol->AddNode(ScifiVolV, 0, new TGeoTranslation(0, 0, 0));
  TGeoBBox* epoxyMatBoxV = new TGeoBBox(Form("%s_epoxyMat", name), width / 2,
                                        height / 2, fiberMatThick / 2);
  TGeoVolume* ScifiMatVolV = new TGeoVolume(
      Form("%s_epoxyMat", name), epoxyMatBoxV, gGeoManager->GetMedium("Epoxy"));
  ScifiMatVolV->SetLineColor(kYellow - 2);
  ScifiMatVolV->SetTransparency(30);
  ScifiMatVolV->SetVisibility(kFALSE);
  ScifiMatVolV->SetVisDaughters(kFALSE);
  ScifiVolV->AddNode(ScifiMatVolV, 0, new TGeoTranslation(0, 0, zFiberMat2));

  // --- Upper Internal Iron ---
  TGeoBBox* upperIronBox = new TGeoBBox(Form("%s_upperIron", name), width / 2,
                                        height / 2, upperIronThick / 2);
  TGeoVolume* upperIronVol = new TGeoVolume(
      Form("%s_upperIron", name), upperIronBox, gGeoManager->GetMedium("iron"));
  upperIronVol->SetLineColor(kGray + 1);
  upperIronVol->SetTransparency(20);
  modMotherVol->AddNode(upperIronVol, 0,
                        new TGeoTranslation(0, 0, zUpperIronInt));

  // -----------------------------
  // Now build the fibers inside each Epoxy block:
  // Common fiber parameters (cm)
  Double_t layerThick = fiberMatThick / numFiberLayers;
  Double_t fiberLength = fFiberLength[iBlock];  // per-block, set in SetMTCParameters
  LOG(info) << "Block " << iBlock << " fiber length set to " << fiberLength
            << " cm";
  Int_t fNumFibers = static_cast<Int_t>(fSciFiActiveX*width/fWidth / fFiberPitch);

  // --- Define the SciFi fiber volume ---
  // Block-unique names: the fibre length differs per block, so these are
  // genuinely distinct volumes and must not share a name across blocks.
  TGeoTube* fiberTube =
      new TGeoTube(Form("FiberTube_%d", iBlock), 0, fFiberRadius, fiberLength / 2);
  TGeoVolume* fiberVol = new TGeoVolume(Form("FiberVol_%d", iBlock), fiberTube,
                                        gGeoManager->GetMedium("SciFiMat"));
  AddSensitiveVolume(fiberVol);
  fiberVol->SetLineColor(kMagenta);
  fiberVol->SetTransparency(15);
  fiberVol->SetVisibility(kFALSE);

  // --- Rotations for U/V fibers ---
  TGeoRotation* rotU = new TGeoRotation();
  rotU->RotateY(fSciFiBendingAngle);
  rotU->RotateX(90.);
  TGeoRotation* rotV = new TGeoRotation();
  rotV->RotateY(-fSciFiBendingAngle);
  rotV->RotateX(90.);

  // --- Place U-fibers inside lower Epoxy ---
  for (int layer = 0; layer < numFiberLayers; ++layer) {
    Double_t z0 = -fiberMatThick / 2 + (layer + 0.5) * (layerThick);
    for (int j = 0; j < fNumFibers; ++j) {
      Double_t x0 = -fSciFiActiveX*width/fWidth / 2 + (j + 0.5) * fFiberPitch;
      if (layer % 2 == 1) {
        if (j == fNumFibers - 1) {
          continue;  // Skip the last layer for odd layers
        }
        x0 += fFiberPitch / 2;
      }
      TGeoCombiTrans* ct = new TGeoCombiTrans("", x0, 0, z0, rotU);
      Int_t copyNo = 100000000 + 1000000 + 0 * 100000 + layer * 10000 + j;
      ScifiMatVolU->AddNode(fiberVol, copyNo, ct);
    }
  }

  // --- Place V-fibers inside upper Epoxy ---
  for (int layer = 0; layer < numFiberLayers; ++layer) {
    Double_t z0 = -fiberMatThick / 2 + (layer + 0.5) * (layerThick);
    for (int j = 0; j < fNumFibers; ++j) {
      Double_t x0 = -fSciFiActiveX*width/fWidth / 2 + (j + 0.5) * fFiberPitch;
      if (layer % 2 == 1) {
        if (j == fNumFibers - 1) {
          continue;  // Skip the last layer for odd layers
        }
        x0 += fFiberPitch / 2;
      }
      TGeoCombiTrans* ct = new TGeoCombiTrans("", x0, 0, z0, rotV);
      Int_t copyNo = 100000000 + 1000000 + 1 * 100000 + layer * 10000 + j;
      ScifiMatVolV->AddNode(fiberVol, copyNo, ct);
    }
  }
}

void MTCDetector::ConstructGeometry() {
  // Initialize media (using FairROOT's interface)
  ShipGeo::InitMedium("SciFiMat");
  ShipGeo::InitMedium("Epoxy");
  ShipGeo::InitMedium("air");
  ShipGeo::InitMedium("iron");
  TGeoMedium* air = gGeoManager->GetMedium("air");
  TGeoMedium* ironMed = gGeoManager->GetMedium("iron");
  // For the scintillator, you may use the same medium as SciFiMat or another if
  // defined.
  TGeoMedium* scintMed = gGeoManager->GetMedium("SciFiMat");
  ShipGeo::InitMedium("silicon");

  // Define the module spacing based on three sublayers:
  //   fIronThick (outer iron), fSciFiThick (SciFi module/fiber module),
  //   fScintThick (scintillator)
  Double_t moduleSpacing = fIronThick + fSciFiThick + fScintThick;
  Double_t totalLength = fLayers * moduleSpacing;


  LOG(info) << "-- Considering " << fnB << " blocks of " << fnLayPerBlock << " layers each.";
  const int nB = fnB;
  // Per-block transverse sizes come from fBlockWidth/fBlockHeight (SetMTCParameters).

  TGeoBBox* ironBox[nB];
  TGeoVolume* ironVol[nB];
  TGeoVolumeAssembly* sensitiveModule[nB];
  auto envBox =
      new TGeoBBox("MTC_env", fWidth / 2, fHeight / 2, totalLength / 2);
  auto envVol = new TGeoVolume("MTC", envBox, air);
  envVol->SetLineColor(kGreen);
  envVol->SetTransparency(50);

  for (int iB(0); iB<nB;++iB){
    // --- Create an envelope volume for the detector (green, semi-transparent)
    // ---
    std::ostringstream label;
    label << "MTC_env_" << iB; 

    label.str("");
    label << "MTC_" << iB;

    // --- Outer Iron Layer (gray) ---
    label.str("");
    label << "MTC_iron_" << iB;
    ironBox[iB] =
      new TGeoBBox(label.str().c_str(), fBlockWidth[iB] / 2, fBlockHeight[iB] / 2, fIronThick / 2);
    label.str("");
    label << "MTC_ironVol_" << iB; 
    ironVol[iB] = new TGeoVolume(label.str().c_str(), ironBox[iB], ironMed);
    ironVol[iB]->SetLineColor(kGray + 1);
    ironVol[iB]->SetTransparency(20);
    // Enable the field in the iron volume
    if (fFieldY != 0) ironVol[iB]->SetField(new TGeoUniformMagField(0, fFieldY, 0));
    
    // --- Assemble the layers into the envelope ---
    label.str("");
    label << "MTC_layer_" << iB; 
    
    sensitiveModule[iB] = new TGeoVolumeAssembly(label.str().c_str());
    // Block-unique name prefix so the per-block volumes/shapes do not collide in
    // the global TGeo registry (e.g. MTC_0_scifi_U vs MTC_1_scifi_U). Built as a
    // std::string (not Form) because the builders call Form() on `name`
    // internally and would otherwise clobber a Form static buffer.
    std::string blockName = "MTC_" + std::to_string(iB);
    // Define a layer for the SciFi module
    CreateSciFiModule(blockName.c_str(), sensitiveModule[iB], fBlockWidth[iB], fBlockHeight[iB], fSciFiThick, iB);
    CreateScintModule(blockName.c_str(), sensitiveModule[iB], fSciFiThick / 2 + fScintThick / 2,
		      fBlockWidth[iB], fBlockHeight[iB], fScintThick, fScintCellSize,
		      fScintCellSize, scintMed, kAzure + 7, 30, iB);
  }

  // Keep overall layer index, access right block index depending on index layer.

  for (Int_t i = 0; i < fLayers; i++) {
    Int_t iB = BlockForLayer(i);  // clamps to [0, fnB-1] if fLayers % fnB != 0
    // Compute the center position (z) for the current module
    Double_t zPos = -totalLength / 2 + i * moduleSpacing;
    
    // Place the Outer Iron layer (shifted down by half the SciFi+scint
    // thickness)
    envVol->AddNode(ironVol[iB], i,
			new TGeoTranslation(0, 0, zPos + fIronThick / 2));
    // Place the sensitive module (SciFi + Scintillator) at the correct z
    // position
    envVol->AddNode(
			sensitiveModule[iB], i,
			new TGeoTranslation(0, 0, zPos + fIronThick + fSciFiThick / 2));
  }
  
  // Finally, add the envelope to the top volume with the global z offset
  // fZCenter

    gGeoManager->GetTopVolume()->AddNode(envVol, 1,
					 new TGeoTranslation(0, 0, fZCenter));

}


Bool_t MTCDetector::ProcessHits(FairVolume* vol) {
  /** This method is called from the MC stepping */
  // Set parameters at entrance of volume. Reset ELoss.
  if (gMC->IsTrackEntering()) {
    fELoss = 0.;
    fTime = gMC->TrackTime() * 1.0e09;
    fLength = gMC->TrackLength();
    gMC->TrackPosition(fPos);
    gMC->TrackMomentum(fMom);
    TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
    Int_t vol_local_id = nav->GetCurrentNode()->GetNumber() %
                         1000000;  // Local ID within the mat or scint.
    Int_t layer_id = nav->GetMother(3)->GetNumber();  // Get layer ID.
    fVolumeID = 100000000 + layer_id * 1000000 +
                vol_local_id;  // 1e8 + layer_id * 1e6 + fibre_local_id;
  }
  // Sum energy loss for all steps in the active volume
  fELoss += gMC->Edep();

  // Create vetoPoint when exiting active volume
  if (gMC->IsTrackExiting() || gMC->IsTrackStop() ||
      gMC->IsTrackDisappeared()) {
    if (fELoss == 0.) {
      return kFALSE;
    }  // if you do not want hits with zero eloss

    TParticle* p = gMC->GetStack()->GetCurrentTrack();
    fTrackID = gMC->GetStack()->GetCurrentTrackNumber();
    Int_t pdgCode = p->GetPdgCode();
    TLorentzVector Pos;
    gMC->TrackPosition(Pos);
    TLorentzVector Mom;
    gMC->TrackMomentum(Mom);
    Double_t x, y, z;
    // 0 and 1 are for SciFi planes, 2 is for scintillating tiles
    if ((fVolumeID / 100000) % 10 == 2) {
      x = (fPos.X() + Pos.X()) / 2.;
      y = (fPos.Y() + Pos.Y()) / 2.;
      z = (fPos.Z() + Pos.Z()) / 2.;
    } else {
      x = fPos.X();
      y = fPos.Y();
      z = (fPos.Z() + Pos.Z()) / 2.;
    }

    AddHit(fTrackID, fVolumeID, TVector3(x, y, z),
           TVector3(fMom.Px(), fMom.Py(), fMom.Pz()),  // entrance momentum
           fTime, fLength, fELoss, pdgCode);
    // hit->Print();
    ShipStack* stack = dynamic_cast<ShipStack*>(gMC->GetStack());
    stack->AddPoint(kMTC);
  }
  return kTRUE;
}

void MTCDetector::SiPMOverlap(Int_t iB){
  // Per-block SiPM channel map. Helper volumes are suffixed with the block
  // index so the three blocks (40/50/60 cm wide) can coexist.
  if (gGeoManager->FindVolumeFast(Form("SiPMmapVolU_%d", iB)) ||
      gGeoManager->FindVolumeFast(Form("SiPMmapVolV_%d", iB))) {
    return;
  }

  double aWidth = fBlockWidth[iB];
  double aHeight = fBlockHeight[iB];

  Double_t fLengthScifiMat = fSciFiActiveY*aHeight/fHeight;
  Double_t fWidthChannel = fFiberPitch * fChannelAggregated;
  Int_t nSiPMChan = std::ceil(aWidth / fWidthChannel);
  Int_t nSiPMs = 1;
  if (nSiPMChan > kMaxChannelsPerSiPM) {
    LOG(warn) << "Number of SiPM channels (" << nSiPMChan
              << ") exceeds maximum per SiPM (" << kMaxChannelsPerSiPM
              << "), redistributing across multiple SiPMs";

    nSiPMs = static_cast<int>(
        std::ceil(static_cast<double>(nSiPMChan) / kMaxChannelsPerSiPM));

    LOG(info) << "Increasing number of SiPMs up to " << nSiPMs;

    // define redistribution of channels among SiPMs
    nSiPMChan =
        static_cast<int>(std::ceil(nSiPMChan / static_cast<float>(nSiPMs)));

    LOG(info) << "New nSiPMChan = " << nSiPMChan;
  }
  fNSiPMChan[iB] = nSiPMChan;
  fNSiPMs[iB] = nSiPMs;
  Double_t fEdge = (aWidth - nSiPMs * nSiPMChan * fWidthChannel) / 2;
  Double_t firstChannelX = -aWidth / 2;

  LOG(info) << "SiPM Overlap parameters (block " << iB << "):\n"
            << "  aWidth = " << aWidth << " cm\n"
            << "  fLengthScifiMat = " << fLengthScifiMat << " cm\n"
            << "  fWidthChannel = " << fWidthChannel << " cm\n"
            << "  fFiberPitch = " << fFiberPitch << " cm\n"
            << "  fChannelAggregated = " << fChannelAggregated << "\n"
            << "  nSiPMChan = " << nSiPMChan << "\n"
            << "  nSiPMs = " << nSiPMs << "\n"
            << "  fNMats = " << fNMats << "\n"
            << "  fEdge = " << fEdge << " cm\n"
            << "  firstChannelX = " << firstChannelX << " cm";
  // Contains all plane SiPMs, defined for horizontal fiber plane
  // To obtain SiPM map for vertical fiber plane rotate by 90 degrees around Z
  TGeoVolumeAssembly* SiPMmapVolU = new TGeoVolumeAssembly(Form("SiPMmapVolU_%d", iB));
  TGeoVolumeAssembly* SiPMmapVolV = new TGeoVolumeAssembly(Form("SiPMmapVolV_%d", iB));

  TGeoBBox* ChannelVol_box = new TGeoBBox(
      Form("ChannelVol_%d", iB), fWidthChannel / 2, fLengthScifiMat / 2, fiberMatThick / 2);
  TGeoVolume* ChannelVol = new TGeoVolume(Form("ChannelVol_%d", iB), ChannelVol_box,
                                          gGeoManager->GetMedium("silicon"));
  /*
    Example of fiberID: 123051820, where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees, 2 for scint plane)
      - 5: z-layer number (0-5)
      - 1820: local fibre ID within the station
    Example of SiPM global channel (what is seen in the output file): 123001123,
    where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees)
      - 0: mat number (only 0 by June 2025)
      - 1: SiPM number (automatically assigned based on fibre aggregation
    settings)
      - 123: number of the SiPM channel (0-N). The channel number depends on the
    fibre aggregation setting.
  */

  Double_t pos = fEdge + firstChannelX;
  for (int imat = 0; imat < fNMats; imat++) {
    for (int isipms = 0; isipms < nSiPMs; isipms++) {
      for (int ichannel = 0; ichannel < nSiPMChan; ichannel++) {
        // +5 degrees
        SiPMmapVolU->AddNode(
            ChannelVol, 100000 * 0 + imat * 10000 + isipms * 1000 + ichannel,
            new TGeoTranslation(pos, 0, 0));
        // -5 degrees
        SiPMmapVolV->AddNode(
            ChannelVol, 100000 * 1 + imat * 10000 + isipms * 1000 + ichannel,
            new TGeoTranslation(-pos, 0, 0));
        pos += fWidthChannel;
      }
    }
  }
}

void MTCDetector::GetPosition(Int_t fDetectorID, TVector3& A, TVector3& B) {
  /*
    Example of fiberID: 123051820, where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees, 2 for scint plane)
      - 5: z-layer number (0-5)
      - 1820: local fibre ID within the station
    Example of SiPM global channel (what is seen in the output file): 123001123,
    where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees)
      - 0: mat number (only 0 by June 2025)
      - 1: SiPM number (automatically assigned based on fibre aggregation
    settings)
      - 123: number of the SiPM channel (0-N). The channel number depends on the
    fibre aggregation setting.
  */

  Int_t station_number = static_cast<int>(fDetectorID / 1e6) % 100;
  Int_t plane_type = static_cast<int>(fDetectorID / 1e5) %
                     10;  // 0 for horizontal, 1 for vertical
  Int_t iB = BlockForLayer(station_number);

  // Fibre copy number is station-independent:
  //   1e8 + 1e6 + plane*1e5 + (layer*1e4 + j), with (layer*1e4 + j) = id % 1e5.
  Int_t fibreCopyNo = 101000000 + plane_type * 100000 + (fDetectorID % 100000);
  // Basic hierarchy (per-block, block-unique volume names):
  // /cave/MTC_1/MTC_layer_<iB>_<station>/MTC_<iB>_scifi_U_0/MTC_<iB>_epoxyMat_0/FiberVol_<iB>_<copyNo>
  TString path = TString::Format("/cave/MTC_1/MTC_layer_%d_%d", iB, station_number);
  path += (plane_type == 0)
              ? TString::Format(
                    "/MTC_%d_scifi_U_0/MTC_%d_epoxyMat_0/FiberVol_%d_%d", iB, iB,
                    iB, fibreCopyNo)
              : TString::Format(
                    "/MTC_%d_scifi_V_0/MTC_%d_epoxyMat_0/FiberVol_%d_%d", iB, iB,
                    iB, fibreCopyNo);
  TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
  nav->cd(path);
  TGeoNode* W = nav->GetCurrentNode();
  TGeoBBox* S = dynamic_cast<TGeoBBox*>(W->GetVolume()->GetShape());

  Double_t top[3] = {0, 0, (S->GetDZ())};
  Double_t bot[3] = {0, 0, -(S->GetDZ())};
  Double_t Gtop[3], Gbot[3];
  nav->LocalToMaster(top, Gtop);
  nav->LocalToMaster(bot, Gbot);
  A.SetXYZ(Gtop[0], Gtop[1], Gtop[2]);
  B.SetXYZ(Gbot[0], Gbot[1], Gbot[2]);
}

TVector3 MTCDetector::GetLocalPos(Int_t fDetectorID, TVector3* glob) {
  Int_t station_number = static_cast<int>(fDetectorID / 1e6) % 100;
  Int_t plane_type = static_cast<int>(fDetectorID / 1e5) %
                     10;  // 0 for horizontal, 1 for vertical
  Int_t iB = BlockForLayer(station_number);

  // Basic hierarchy:
  // /cave/MTC_1/MTC_layer_<iB>_<station>/MTC_<iB>_scifi_U_0
  TString path = TString::Format("/cave/MTC_1/MTC_layer_%d_%d", iB, station_number);
  path += (plane_type == 0) ? TString::Format("/MTC_%d_scifi_U_0", iB)
                            : TString::Format("/MTC_%d_scifi_V_0", iB);
  TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
  nav->cd(path);
  Double_t aglob[3];
  Double_t aloc[3];
  glob->GetXYZ(aglob);
  nav->MasterToLocal(aglob, aloc);
  return TVector3(aloc[0], aloc[1], aloc[2]);
}

void MTCDetector::GetSiPMPosition(Int_t SiPMChan, TVector3& A, TVector3& B) {
  /* STMRFFF
      Example of fiberID: 123051820, where:
        - 1: MTC unique ID
        - 23: layer number
        - 0: station type (0 for +5 degrees, 1 for -5 degrees, 2 for scint
     plane)
        - 5: z-layer number (0-5)
        - 1820: local fibre ID within the station
      Example of SiPM global channel (what is seen in the output file):
     123001123, where:
        - 1: MTC unique ID
        - 23: layer number
        - 0: station type (0 for +5 degrees, 1 for -5 degrees)
        - 0: mat number (only 0 by June 2025)
        - 1: SiPM number (automatically assigned based on fibre aggregation
     settings)
        - 123: number of the SiPM channel (0-N). The channel number depends on
     the fibre aggregation setting.
  */
  Int_t locNumber = SiPMChan % 1000000;
  Int_t station_number = static_cast<int>(SiPMChan / 1e6) % 100;
  Int_t plane_type = static_cast<int>(SiPMChan / 1e5) %
                     10;  // 0 for horizontal, 1 for vertical
  Int_t iB = BlockForLayer(station_number);
  Float_t locPosition;
  locPosition = (plane_type == 0 ? SiPMPos_U[iB]
                                 : SiPMPos_V[iB])[locNumber];  // local U/V pos

  Double_t loc[3] = {0, 0, 0};
  TString path = TString::Format("/cave/MTC_1/MTC_layer_%d_%d", iB, station_number);
  path += (plane_type == 0)
              ? TString::Format("/MTC_%d_scifi_U_0/MTC_%d_epoxyMat_0", iB, iB)
              : TString::Format("/MTC_%d_scifi_V_0/MTC_%d_epoxyMat_0", iB, iB);
  TGeoNavigator* nav = gGeoManager->GetCurrentNavigator();
  Double_t glob[3] = {0, 0, 0};
  loc[0] = locPosition;
  loc[1] = -fFiberLength[iB] / 2;
  loc[2] = 7.47;
  nav->cd(path);
  nav->LocalToMaster(loc, glob);
  A.SetXYZ(glob[0], glob[1], glob[2]);
  loc[0] = locPosition;
  loc[1] = fFiberLength[iB] / 2;
  loc[2] = 7.47;  // hardcoded for now, for some reason required to get the
                  // correct local position
  nav->LocalToMaster(loc, glob);
  B.SetXYZ(glob[0], glob[1], glob[2]);
}

void MTCDetector::SiPMmapping(){
  // check if containers are already filled
  if (!SiPMPos_U.empty() && !SiPMPos_U[0].empty()) {
    LOG(warning) << "SiPM mapping already done, skipping.";
    return;
  }

  // Fetch a daughter volume of `mother` whose node name contains `part`.
  auto daughterVol = [](TGeoVolume* mother, const char* part) -> TGeoVolume* {
    if (!mother || !mother->GetNodes()) return nullptr;
    TIter next(mother->GetNodes());
    TGeoNode* nd;
    while ((nd = static_cast<TGeoNode*>(next()))) {
      if (TString(nd->GetName()).Contains(part)) return nd->GetVolume();
    }
    return nullptr;
  };

  // One transverse map per block: blocks differ in width (SiPM channel count)
  // and height (fibre length), so a fibre-local ID means a different physical
  // position in each block.
  for (Int_t iB = 0; iB < fnB; ++iB) {
    SiPMOverlap(iB);
    // Fibre transverse positions are identical for every station within a
    // block, so a single representative station is enough to build the map.
    Int_t repStation = iB * fnLayPerBlock;
    // The per-block sensitive-module assembly is uniquely named, so navigating
    // from it disambiguates the (duplicated) inner SciFi volume names.
    auto layerVol = gGeoManager->FindVolumeFast(Form("MTC_layer_%d", iB));
    if (!layerVol) {
      LOG(warning) << "MTC_layer_" << iB << " not found, skipping block " << iB;
      continue;
    }

    for (int plane = 0; plane < 2; ++plane) {  // 0 = U, 1 = V
      const char* sipmName = (plane == 0) ? "SiPMmapVolU" : "SiPMmapVolV";
      const char* scifiPart = (plane == 0) ? "scifi_U" : "scifi_V";
      auto& fibresSiPM = (plane == 0) ? fibresSiPM_U[iB] : fibresSiPM_V[iB];
      auto& siPMFibres = (plane == 0) ? siPMFibres_U[iB] : siPMFibres_V[iB];
      auto& SiPMPos = (plane == 0) ? SiPMPos_U[iB] : SiPMPos_V[iB];

      auto sipm = gGeoManager->FindVolumeFast(Form("%s_%d", sipmName, iB));
      if (!sipm || !sipm->GetNodes()) continue;
      TObjArray* Nodes = sipm->GetNodes();
      auto planeVol = daughterVol(layerVol, scifiPart);
      if (!planeVol || !planeVol->GetNodes()) continue;

      Float_t fibresRadius = -1;
      Float_t dSiPM = -1;
      for (int imat = 0; imat < planeVol->GetNodes()->GetEntries(); imat++) {
        auto mat = static_cast<TGeoNode*>(planeVol->GetNodes()->At(imat));
        auto vmat = mat->GetVolume();
        for (int ifibre = 0; ifibre < vmat->GetNodes()->GetEntriesFast();
             ifibre++) {
          auto fibre = static_cast<TGeoNode*>(vmat->GetNodes()->At(ifibre));
          if (fibresRadius < 0) {
            auto S = dynamic_cast<TGeoBBox*>(fibre->GetVolume()->GetShape());
            fibresRadius = S->GetDX();
          }
          Int_t fID = fibre->GetNumber() % 1000000 +
                      imat * 1e4;  // local fibre number within the block plane
          // Fibre copy numbers do not encode the station; inject the block's
          // representative station so GetPosition navigates to this block.
          Int_t globalFibreID =
              100000000 + repStation * 1000000 + fibre->GetNumber() % 1000000;
          TVector3 Atop, Bbot;
          GetPosition(globalFibreID, Atop, Bbot);
          Float_t a = Bbot[0];

          //  check for overlap with any of the SiPM channels in the same mat
          for (Int_t nChan = 0; nChan < Nodes->GetEntriesFast(); nChan++) {
            auto vol = static_cast<TGeoNode*>(Nodes->At(nChan));
            Int_t N = vol->GetNumber();
            Float_t xcentre = vol->GetMatrix()->GetTranslation()[0];
            if (dSiPM < 0) {
              TGeoBBox* B =
                  dynamic_cast<TGeoBBox*>(vol->GetVolume()->GetShape());
              dSiPM = B->GetDX();
            }

            if (TMath::Abs(xcentre - a) > 3 * dSiPM / 2) {
              continue;
            }  // no need to check further
            Float_t W = area(a, fibresRadius, xcentre - dSiPM, xcentre + dSiPM);
            if (W < 0) {
              continue;
            }
            std::array<float, 2> Wa;
            Wa[0] = W;
            Wa[1] = a;
            fibresSiPM[N][fID] = Wa;
          }
        }
      }
      // calculate also local SiPM positions based on fibre positions and their
      // fraction probably an overkill, maximum difference between weighted
      // average and central position < 6 micron.
      for (auto& [N, it] : fibresSiPM) {
        Float_t m = 0;
        Float_t w = 0;
        for (auto& [current_fibre, Wa] : it) {
          m += Wa[0] * Wa[1];
          w += Wa[0];
        }
        SiPMPos[N] = m / w;
      }
      // make inverse mapping, which fibre is associated to which SiPMs
      for (auto& [N, it] : fibresSiPM) {
        for (auto& [nfibre, Wa] : it) {
          siPMFibres[nfibre][N] = Wa;
        }
      }
    }
  }
}
