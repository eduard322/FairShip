#include "ShipMuonShield.h"

#include "TGeoManager.h"
#include "TObjArray.h"                  // for TObjArray
#include "TString.h"                    // for TString
#include "TGeoBBox.h"
#include "TGeoTrd1.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoTube.h"
#include "TGeoMaterial.h"
#include "FairGeoInterface.h"
#include "FairGeoMedia.h"
#include "FairGeoBuilder.h"
#include "FairRuntimeDb.h"              // for FairRuntimeDb
#include "TFile.h"
#include <iostream>                     // for operator<<, basic_ostream, etc

Double_t cm = 1;
Double_t m = 100 * cm;
Double_t mm = 0.1 * cm;
Double_t kilogauss = 1.;
Double_t tesla = 10 * kilogauss;

ShipMuonShield::~ShipMuonShield() {}
ShipMuonShield::ShipMuonShield() : FairModule("ShipMuonShield", "") {}

ShipMuonShield::ShipMuonShield(TString geofile,
                               Double_t floor,
                               const Int_t withCoMagnet, const Bool_t StepGeo,
                               const Bool_t WithConstAbsorberField, const Bool_t WithConstShieldField)
  : FairModule("MuonShield", "ShipMuonShield")
{
  fWithConstAbsorberField = WithConstAbsorberField;
  fWithConstShieldField = WithConstShieldField;
  fStepGeo = StepGeo;
  fWithCoMagnet = withCoMagnet;
  fGeofile = geofile;
  auto f = TFile::Open(geofile, "read");
  TVectorT<Double_t> params;
  params.Read("params");
  Double_t LE = 7 * m;
  fDesign = 8;
  fSC_mag = false;
  fField = 1.7;
  dZ0 = 1 * m;
  dZ1 = 0.4 * m;
  dZ2 = 2.31 * m;
  dZ3 = params[2];
  dZ4 = params[3];
  dZ5 = params[4];
  dZ6 = params[5];
  dZ7 = params[6];
  dZ8 = params[7];
  fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7 + dZ8) + LE;

  fFloor = floor;
  fSupport = false;

  Double_t Z = -25 * m - fMuonShieldLength / 2.;

  zEndOfAbsorb = Z + - fMuonShieldLength / 2.;
}

ShipMuonShield::ShipMuonShield(TVectorT<Double_t> in_params,
Double_t floor, const Int_t withCoMagnet, const Bool_t StepGeo, const Bool_t WithConstAbsorberField, const Bool_t WithConstShieldField)
  : FairModule("MuonShield", "ShipMuonShield")
{
  for(int i = 0; i < 56; i++){
      shield_params.push_back(in_params[i]);
  }
  fWithConstAbsorberField = WithConstAbsorberField;
  fWithConstShieldField = WithConstShieldField;
  fStepGeo = StepGeo;
  fWithCoMagnet = withCoMagnet;
  Double_t LE = 7 * m;
  fDesign = 8;
  fSC_mag = true;
  fField = 1.7;
  dZ0 = 1 * m;
  dZ1 = 0.4 * m;
  dZ2 = 2.31 * m;
  dZ3 = in_params[2];
  dZ4 = in_params[3];
  dZ5 = in_params[4];
  dZ6 = in_params[5];
  dZ7 = in_params[6];
  dZ8 = in_params[7];
  fMuonShieldLength = 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7 + dZ8) + LE;

  fFloor = floor;
  fSupport = false;

  Double_t Z = -25 * m - fMuonShieldLength / 2.;

  zEndOfAbsorb = Z + - fMuonShieldLength / 2.;
}

ShipMuonShield::ShipMuonShield(const char* name, const Int_t Design, const char* Title,
                               Double_t Z, Double_t L0, Double_t L1, Double_t L2, Double_t L3, Double_t L4, Double_t L5, Double_t L6,
                               Double_t L7, Double_t L8, Double_t gap, Double_t LE, Double_t, Double_t floor, Double_t field,
                               const Int_t withCoMagnet, const Bool_t StepGeo,
                               const Bool_t WithConstAbsorberField, const Bool_t WithConstShieldField)
  : FairModule(name ,Title)
{
 fDesign = Design;
 fField  = field;
 fSC_mag = false;
 fGeofile = "";
 fWithConstAbsorberField = WithConstAbsorberField;
 fWithConstShieldField = WithConstShieldField;
 fStepGeo = StepGeo;
 fWithCoMagnet = withCoMagnet;
 if (fDesign < 7){
     Fatal("ShipMuonShield","Design %i not anymore supported",fDesign);
 } else if (fDesign >= 7){
     dZ1 = L1;
     dZ2 = L2;
     dZ3 = L3;
     dZ4 = L4;
     dZ5 = L5;
     dZ6 = L6;
     dZ7 = L7;
     dZ8 = L8;
     fMuonShieldLength =
	 2 * (dZ1 + dZ2 + dZ3 + dZ4 + dZ5 + dZ6 + dZ7 + dZ8) + LE;
 }

 fFloor = floor;

 zEndOfAbsorb = Z - fMuonShieldLength/2.;

 fSupport = false;
}

// -----   Private method InitMedium
Int_t ShipMuonShield::InitMedium(TString name)
{
   static FairGeoLoader *geoLoad=FairGeoLoader::Instance();
   static FairGeoInterface *geoFace=geoLoad->getGeoInterface();
   static FairGeoMedia *media=geoFace->getMedia();
   static FairGeoBuilder *geoBuild=geoLoad->getGeoBuilder();

   FairGeoMedium *ShipMedium=media->getMedium(name);

   if (!ShipMedium)
     Fatal("InitMedium","Material %s not defined in media file.", name.Data());
   TGeoMedium* medium=gGeoManager->GetMedium(name);
   if (medium)
     return ShipMedium->getMediumIndex();
   return geoBuild->createMedium(ShipMedium);
}

void ShipMuonShield::CreateTube(TString tubeName, TGeoMedium *medium,
				Double_t dX, Double_t dY, Double_t dZ,
				Int_t color, TGeoVolume *tShield,
				Double_t x_translation, Double_t y_translation,
				Double_t z_translation) {
  TGeoVolume *absorber = gGeoManager->MakeTube(tubeName, medium, dX, dY, dZ);
  absorber->SetLineColor(color);
  tShield->AddNode(
      absorber, 1,
      new TGeoTranslation(x_translation, y_translation, z_translation));
}

void ShipMuonShield::CreateArb8(TString arbName, TGeoMedium *medium,
				Double_t dZ, std::array<Double_t, 16> corners,
				Int_t color, TGeoUniformMagField *magField,
				TGeoVolume *tShield, Double_t x_translation,
				Double_t y_translation,
				Double_t z_translation) {
  TGeoVolume *magF =
      gGeoManager->MakeArb8(arbName, medium, dZ, corners.data());
  magF->SetLineColor(color);
  if (arbName.Contains("Absorb")) {
      if (fWithConstAbsorberField) {
          magF->SetField(magField);
      }
  } else if (fWithConstShieldField) {
      magF->SetField(magField);
  }
  tShield->AddNode(magF, 1, new TGeoTranslation(x_translation, y_translation,
						z_translation));
}

void ShipMuonShield::CreateArb8(TString arbName, TGeoMedium *medium,
          Double_t dZ, std::array<Double_t, 16> corners,
          Int_t color, TGeoUniformMagField *magField,
          TGeoVolume *tShield, Double_t x_translation,
          Double_t y_translation,
          Double_t z_translation,
          Bool_t stepGeo) {
  if (!stepGeo)
  {
    CreateArb8 (arbName, medium, dZ, corners, color, magField, tShield, x_translation, y_translation, z_translation);
    return;
  }
  Double_t partLength = 0.5;
  Int_t zParts = std::ceil(2.0*dZ/m/partLength);
  Double_t finalCorners[zParts][16];
  Double_t dxdy[4][2];
  Double_t dZp = dZ/Double_t(zParts);

  for (int i = 0; i < 4; ++i)
  {
    dxdy[i][0] = (corners[8+2*i] - corners[0+2*i])/Double_t(zParts);
    dxdy[i][1] = (corners[9+2*i] - corners[1+2*i])/Double_t(zParts);
  }

  std::copy(corners.data() + 0,  corners.data() + 8, finalCorners[0]);

  for (int i = 0; i < zParts; ++i)
  {
    for (int k = 0; k < 4; ++k)
    {
      finalCorners[i][8+2*k] = finalCorners[i][0+2*k] + dxdy[k][0];
      finalCorners[i][9+2*k] = finalCorners[i][1+2*k] + dxdy[k][1];
    }
    if (i != zParts-1)
    {
      std::copy(finalCorners[i] + 8, finalCorners[i] + 16, finalCorners[i+1]);
    }
  }

  for (int i = 0; i < zParts; ++i)
  {
    for (int k = 0; k < 4; ++k)
    {
      finalCorners[i][8+2*k] = finalCorners[i][0+2*k]  = (finalCorners[i][0+2*k] + finalCorners[i][8+2*k]) / 2.0;
      finalCorners[i][9+2*k] = finalCorners[i][1+2*k]  = (finalCorners[i][9+2*k] + finalCorners[i][1+2*k]) / 2.0;
    }
  }

  std::vector<TGeoVolume*> magF;

  for (int i = 0; i < zParts; ++i)
  {
    magF.push_back(gGeoManager->MakeArb8(arbName + '_' + std::to_string(i), medium, dZp - 0.00001*m, finalCorners[i]));
    magF[i]->SetLineColor(color);
    if (fWithConstShieldField) {
      magF[i]->SetField(magField);
    }
  }

  for (int i = 0; i < zParts; ++i)
  {
    Double_t true_z_translation = z_translation + 2.0 * Double_t(i) * dZp - dZ + dZp;
    tShield->AddNode(magF[i], 1, new TGeoTranslation(x_translation, y_translation, true_z_translation));
  }
}

void ShipMuonShield::CreateMagnet(TString magnetName,TGeoMedium* medium,TGeoVolume *tShield,TGeoUniformMagField *fields[4],FieldDirection fieldDirection,
				  Double_t dX, Double_t dY, Double_t dX2, Double_t dY2, Double_t dZ,
				  Double_t middleGap,Double_t middleGap2,
				  Double_t HmainSideMag, Double_t HmainSideMag2,
				  Double_t gap,Double_t gap2, Double_t Z, Bool_t NotMagnet,
          Bool_t stepGeo, Bool_t SC_key = false)
  {
    Double_t coil_gap,coil_gap2;
    Int_t color[4] = {45,31,30,38};

    if (NotMagnet) {
       coil_gap = gap;
       coil_gap2 = gap2;
    } else if (fDesign > 7) {
       // Assuming 0.5A/mm^2 and 10000At needed, about 200cm^2 gaps are necessary
       // Current design safely above this. Will consult with MISiS to get a better minimum.
       gap = std::ceil(std::max(100. / dY, gap));
       gap2 = std::ceil(std::max(100. / dY2, gap2));
       coil_gap = gap;
       coil_gap2 = gap2;
    } else {
       coil_gap = std::max(20., gap);
       coil_gap2 = std::max(20., gap2);
       gap = std::max(2., gap);
       gap2 = std::max(2., gap2);
    }

    Double_t anti_overlap = 0.1; // gap between fields in the
						   // corners for mitred joints
						   // (Geant goes crazy when
						   // they touch each other)

    std::array<Double_t, 16> cornersMainL = {
	middleGap, -(dY + dX - anti_overlap),
	middleGap, dY + dX - anti_overlap,
	dX + middleGap, dY - anti_overlap,
	dX + middleGap, -(dY - anti_overlap),
	middleGap2, -(dY2 + dX2 - anti_overlap),
	middleGap2, dY2 + dX2 - anti_overlap,
	dX2 + middleGap2, dY2 - anti_overlap,
	dX2 + middleGap2, -(dY2 - anti_overlap)
    };

    std::array<Double_t, 16> cornersTL = {middleGap + dX,
                                          dY,
                                          middleGap,
                                          dY + dX,
                                          2 * dX + middleGap + coil_gap,
                                          dY + dX,
                                          dX + middleGap + coil_gap,
                                          dY,
                                          middleGap2 + dX2,
                                          dY2,
                                          middleGap2,
                                          dY2 + dX2,
                                          2 * dX2 + middleGap2 + coil_gap2,
                                          dY2 + dX2,
                                          dX2 + middleGap2 + coil_gap2,
                                          dY2};

    std::array<Double_t, 16> cornersMainSideL =
      fDesign == 7 ?
      std::array<Double_t, 16>{
	dX + middleGap + gap, -HmainSideMag,
	dX + middleGap + gap, HmainSideMag,
	2 * dX + middleGap + gap, HmainSideMag,
	2 * dX + middleGap + gap, -HmainSideMag,
	dX2 + middleGap2 + gap2, -HmainSideMag2,
	dX2 + middleGap2 + gap2, HmainSideMag2,
	2 * dX2 + middleGap2 + gap2, HmainSideMag2,
	2 * dX2 + middleGap2 + gap2, -HmainSideMag2
      } :
      std::array<Double_t, 16>{
	dX + middleGap + gap, -(dY - anti_overlap),
	dX + middleGap + gap, dY - anti_overlap,
	2 * dX + middleGap + gap, dY + dX - anti_overlap,
	2 * dX + middleGap + gap, -(dY + dX - anti_overlap),
	dX2 + middleGap2 + gap2, -(dY2 - anti_overlap),
	dX2 + middleGap2 + gap2, dY2 - anti_overlap,
	2 * dX2 + middleGap2 + gap2, dY2 + dX2 - anti_overlap,
	2 * dX2 + middleGap2 + gap2, -(dY2 + dX2 - anti_overlap)
    };
    // SC part
  if(SC_key){

    cornersMainL = {
    middleGap, -(dY + 3*dX - anti_overlap),
    middleGap, dY + 3*dX - anti_overlap,
    dX + middleGap, dY - anti_overlap,
    dX + middleGap, -(dY - anti_overlap),
    middleGap2, -(dY2 + 3*dX2 - anti_overlap),
    middleGap2, dY2 + 3*dX2 - anti_overlap,
    dX2 + middleGap2, dY2 - anti_overlap,
    dX2 + middleGap2, -(dY2 - anti_overlap)
      };

      cornersTL = {middleGap + dX,
                  dY,
                  middleGap,
                  dY + 3*dX,
                  4 * dX + middleGap + coil_gap,
                  dY + 3*dX,
                  dX + middleGap + coil_gap,
                  dY,
                  middleGap2 + dX2,
                  dY2,
                  middleGap2,
                  dY2 + 3*dX2,
                  4 * dX2 + middleGap2 + coil_gap2,
                  dY2 + 3*dX2,
                  dX2 + middleGap2 + coil_gap2,
                  dY2};

      cornersMainSideL =
        fDesign == 7 ?
        std::array<Double_t, 16>{
    dX + middleGap + gap, -HmainSideMag,
    dX + middleGap + gap, HmainSideMag,
    4 * dX + middleGap + gap, HmainSideMag,
    4 * dX + middleGap + gap, -HmainSideMag,
    dX2 + middleGap2 + gap2, -HmainSideMag2,
    dX2 + middleGap2 + gap2, HmainSideMag2,
    4 * dX2 + middleGap2 + gap2, HmainSideMag2,
    4 * dX2 + middleGap2 + gap2, -HmainSideMag2
        } :
        std::array<Double_t, 16>{
    dX + middleGap + gap, -(dY - anti_overlap),
    dX + middleGap + gap, dY - anti_overlap,
    4 * dX + middleGap + gap, dY + 3*dX - anti_overlap,
    4 * dX + middleGap + gap, -(dY + 3*dX - anti_overlap),
    dX2 + middleGap2 + gap2, -(dY2 - anti_overlap),
    dX2 + middleGap2 + gap2, dY2 - anti_overlap,
    4 * dX2 + middleGap2 + gap2, dY2 + 3*dX2 - anti_overlap,
    4 * dX2 + middleGap2 + gap2, -(dY2 + 3*dX2 - anti_overlap)
    };
  }
      // END of SC part
    std::array<Double_t, 16> cornersMainR, cornersCLBA,
       cornersMainSideR, cornersCLTA, cornersCRBA,
       cornersCRTA, cornersTR, cornersBL, cornersBR;

    if (fDesign == 7) {
       cornersCLBA = {dX + middleGap + gap,
                      -HmainSideMag,
                      2 * dX + middleGap + gap,
                      -HmainSideMag,
                      2 * dX + middleGap + coil_gap,
                      -(dY + dX - anti_overlap),
                      dX + middleGap + coil_gap,
                      -(dY - anti_overlap),
                      dX2 + middleGap2 + gap2,
                      -HmainSideMag2,
                      2 * dX2 + middleGap2 + gap2,
                      -HmainSideMag2,
                      2 * dX2 + middleGap2 + coil_gap2,
                      -(dY2 + dX2 - anti_overlap),
                      dX2 + middleGap2 + coil_gap2,
                      -(dY2 - anti_overlap)};
    }

    // Use symmetries to define remaining magnets
    for (int i = 0; i < 16; ++i) {
      cornersMainR[i] = -cornersMainL[i];
      cornersMainSideR[i] = -cornersMainSideL[i];
      cornersCRTA[i] = -cornersCLBA[i];
      cornersBR[i] = -cornersTL[i];
    }
    // Need to change order as corners need to be defined clockwise
    for (int i = 0, j = 4; i < 8; ++i) {
      j = (11 - i) % 8;
      cornersCLTA[2 * j] = cornersCLBA[2 * i];
      cornersCLTA[2 * j + 1] = -cornersCLBA[2 * i + 1];
      cornersTR[2 * j] = -cornersTL[2 * i];
      cornersTR[2 * j + 1] = cornersTL[2 * i + 1];
    }
    for (int i = 0; i < 16; ++i) {
      cornersCRBA[i] = -cornersCLTA[i];
      cornersBL[i] = -cornersTR[i];
    }

    TString str1L = "_MiddleMagL";
    TString str1R = "_MiddleMagR";
    TString str2 = "_MagRetL";
    TString str3 = "_MagRetR";
    TString str4 = "_MagCLB";
    TString str5 = "_MagCLT";
    TString str6 = "_MagCRT";
    TString str7 = "_MagCRB";
    TString str8 = "_MagTopLeft";
    TString str9 = "_MagTopRight";
    TString str10 = "_MagBotLeft";
    TString str11 = "_MagBotRight";

    switch (fieldDirection){

    case FieldDirection::up:
      CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, color[0], fields[0], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, color[0], fields[0], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, color[1], fields[1], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, color[1], fields[1], tShield,  0, 0, Z, stepGeo);
      if (fDesign == 7) {
         CreateArb8(magnetName + str4, medium, dZ, cornersCLBA, color[1], fields[1], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str5, medium, dZ, cornersCLTA, color[1], fields[1], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str6, medium, dZ, cornersCRTA, color[1], fields[1], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str7, medium, dZ, cornersCRBA, color[1], fields[1], tShield, 0, 0, Z, stepGeo);
      }
      CreateArb8(magnetName + str8, medium, dZ, cornersTL, color[3], fields[3], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str9, medium, dZ, cornersTR, color[2], fields[2], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str10, medium, dZ, cornersBL, color[2], fields[2], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str11, medium, dZ, cornersBR, color[3], fields[3], tShield,  0, 0, Z, stepGeo);
      break;
    case FieldDirection::down:
      CreateArb8(magnetName + str1L, medium, dZ, cornersMainL, color[1], fields[1], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str1R, medium, dZ, cornersMainR, color[1], fields[1], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str2, medium, dZ, cornersMainSideL, color[0], fields[0], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str3, medium, dZ, cornersMainSideR, color[0], fields[0], tShield,  0, 0, Z, stepGeo);
      if (fDesign == 7) {
         CreateArb8(magnetName + str4, medium, dZ, cornersCLBA, color[0], fields[0], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str5, medium, dZ, cornersCLTA, color[0], fields[0], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str6, medium, dZ, cornersCRTA, color[0], fields[0], tShield, 0, 0, Z, stepGeo);
         CreateArb8(magnetName + str7, medium, dZ, cornersCRBA, color[0], fields[0], tShield, 0, 0, Z, stepGeo);
      }
      CreateArb8(magnetName + str8, medium, dZ, cornersTL, color[2], fields[2], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str9, medium, dZ, cornersTR, color[3], fields[3], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str10, medium, dZ, cornersBL, color[3], fields[3], tShield,  0, 0, Z, stepGeo);
      CreateArb8(magnetName + str11, medium, dZ, cornersBR, color[2], fields[2], tShield,  0, 0, Z, stepGeo);
      break;
    }
  }

Int_t ShipMuonShield::Initialize(std::vector<TString> &magnetName,
				std::vector<FieldDirection> &fieldDirection,
				std::vector<Double_t> &dXIn, std::vector<Double_t> &dYIn,
				std::vector<Double_t> &dXOut, std::vector<Double_t> &dYOut,
				std::vector<Double_t> &dZ, std::vector<Double_t> &midGapIn,
				std::vector<Double_t> &midGapOut,
				std::vector<Double_t> &HmainSideMagIn,
				std::vector<Double_t> &HmainSideMagOut,
				std::vector<Double_t> &gapIn, std::vector<Double_t> &gapOut,
				std::vector<Double_t> &Z) {

  const Int_t nMagnets = 9;
  magnetName.reserve(nMagnets);
  fieldDirection.reserve(nMagnets);
  for (auto i :
       {&dXIn, &dXOut, &dYIn, &dYOut, &dZ, &midGapIn, &midGapOut,
	&HmainSideMagIn, &HmainSideMagOut, &gapIn, &gapOut, &Z}) {
    i->reserve(nMagnets);
  }

  Double_t zgap = 10 * cm;  // fixed distance between magnets in Z-axis

  if (fDesign == 8) {

    magnetName = {"MagnAbsorb1", "MagnAbsorb2", "Magn1", "Magn2", "Magn3",
		  "Magn4",       "Magn5",       "Magn6", "Magn7"};

    fieldDirection = {
	FieldDirection::up,   FieldDirection::up,   FieldDirection::up,
	FieldDirection::up,   FieldDirection::up,   FieldDirection::down,
	FieldDirection::down, FieldDirection::down, FieldDirection::down,
    };

    std::vector<Double_t> params;
    if (!fSC_mag)
    {
      TVectorT<Double_t> paramsR;
      auto f = TFile::Open(fGeofile, "read");
      paramsR.Read("params");
      for (int i = 0; i < 56; i++){params.push_back(paramsR[i]);}
    }else{
      params = shield_params;
    }

    const int offset = 7;

    dXIn[0] = 0.4 * m;
    dXOut[0] = 0.40 * m;
    gapIn[0] = 0.1 * mm;
    dYIn[0] = 1.5 * m;
    dYOut[0] = 1.5 * m;
    gapOut[0] = 0.1 * mm;
    dXIn[1] = 0.5 * m;
    dXOut[1] = 0.5 * m;
    gapIn[1] = 0.02 * m;
    dYIn[1] = 1.19 * m;
    dYOut[1] = 1.19 * m;
    gapOut[1] = 0.02 * m;

    for (Int_t i = 2; i < nMagnets - 1; ++i) {
      dXIn[i] = params[offset + i * 6 + 1];
      dXOut[i] = params[offset + i * 6 + 2];
      dYIn[i] = params[offset + i * 6 + 3];
      dYOut[i] = params[offset + i * 6 + 4];
      gapIn[i] = params[offset + i * 6 + 5];
      gapOut[i] = params[offset + i * 6 + 6];
    }

    dZ[0] = dZ1 - zgap / 2;
    Z[0] = zEndOfAbsorb + dZ[0] + zgap;
    dZ[1] = dZ2 - zgap / 2;
    Z[1] = Z[0] + dZ[0] + dZ[1] + zgap;
    dZ[2] = dZ3 - zgap / 2;
    Z[2] = Z[1] + dZ[1] + dZ[2] + 2 * zgap;
    dZ[3] = dZ4 - zgap / 2;
    Z[3] = Z[2] + dZ[2] + dZ[3] + zgap;
    dZ[4] = dZ5 - zgap / 2;
    Z[4] = Z[3] + dZ[3] + dZ[4] + zgap;
    dZ[5] = dZ6 - zgap / 2;
    Z[5] = Z[4] + dZ[4] + dZ[5] + zgap;
    dZ[6] = dZ7 - zgap / 2;
    Z[6] = Z[5] + dZ[5] + dZ[6] + zgap;
    dZ[7] = dZ8 - zgap / 2;
    Z[7] = Z[6] + dZ[6] + dZ[7] + zgap;

    dXIn[8] = dXOut[7];
    dYIn[8] = dYOut[7];
    dXOut[8] = dXIn[8];
    dYOut[8] = dYIn[8];
    gapIn[8] = gapOut[7];
    gapOut[8] = gapIn[8];
    dZ[8] = 0.1 * m;
    Z[8] = Z[7] + dZ[7] + dZ[8];

    for (int i = 0; i < nMagnets; ++i) {
      midGapIn[i] = 0.;
      midGapOut[i] = 0.;
      HmainSideMagIn[i] = dYIn[i] / 2;
      HmainSideMagOut[i] = dYOut[i] / 2;
    }

  } else if (fDesign == 9) {
     magnetName = {"MagnAbsorb1", "MagnAbsorb2", "Magn1", "Magn2", "Magn3",
       "Magn4", "Magn5", "Magn6", "Magn7"
     };

     fieldDirection = {
        FieldDirection::up, FieldDirection::up, FieldDirection::up,
	FieldDirection::up, FieldDirection::up, FieldDirection::down,
	FieldDirection::down, FieldDirection::down, FieldDirection::down,
     };

     dXIn[0] = 0.4 * m;
     dXOut[0] = 0.40 * m;
     dYIn[0] = 1.5 * m;
     dYOut[0] = 1.5 * m;
     gapIn[0] = 0.1 * mm;
     gapOut[0] = 0.1 * mm;
     dZ[0] = dZ1 - zgap / 2;
     Z[0] = zEndOfAbsorb + dZ[0] + zgap;

     dXIn[1] = 0.5 * m;
     dXOut[1] = 0.5 * m;
     dYIn[1] = 1.3 * m;
     dYOut[1] = 1.3 * m;
     gapIn[1] = 0.02 * m;
     gapOut[1] = 0.02 * m;
     dZ[1] = dZ2 - zgap / 2;
     Z[1] = Z[0] + dZ[0] + dZ[1] + zgap;

     dXIn[2] = 0.72 * m;
     dXOut[2] = 0.51 * m;
     dYIn[2] = 0.29 * m;
     dYOut[2] = 0.46 * m;
     gapIn[2] = 0.10 * m;
     gapOut[2] = 0.07 * m;
     dZ[2] = dZ3 - zgap / 2;
     Z[2] = Z[1] + dZ[1] + dZ[2] + 2 * zgap;

     dXIn[3] = 0.54 * m;
     dXOut[3] = 0.38 * m;
     dYIn[3] = 0.46 * m;
     dYOut[3] = 1.92 * m;
     gapIn[3] = 0.14 * m;
     gapOut[3] = 0.09 * m;
     dZ[3] = dZ4 - zgap / 2;
     Z[3] = Z[2] + dZ[2] + dZ[3] + zgap;

     dXIn[4] = 0.10 * m;
     dXOut[4] = 0.31 * m;
     dYIn[4] = 0.35 * m;
     dYOut[4] = 0.31 * m;
     gapIn[4] = 0.51 * m;
     gapOut[4] = 0.11 * m;
     dZ[4] = dZ5 - zgap / 2;
     Z[4] = Z[3] + dZ[3] + dZ[4] + zgap;

     dXIn[5] = 0.03 * m;
     dXOut[5] = 0.32 * m;
     dYIn[5] = 0.54 * m;
     dYOut[5] = 0.24 * m;
     gapIn[5] = 0.08 * m;
     gapOut[5] = 0.08 * m;
     dZ[5] = dZ6 - zgap / 2;
     Z[5] = Z[4] + dZ[4] + dZ[5] + zgap;

     dXIn[6] = 0.22 * m;
     dXOut[6] = 0.32 * m;
     dYIn[6] = 2.09 * m;
     dYOut[6] = 0.35 * m;
     gapIn[6] = 0.08 * m;
     gapOut[6] = 0.13 * m;
     dZ[6] = dZ7 - zgap / 2;
     Z[6] = Z[5] + dZ[5] + dZ[6] + zgap;

     dXIn[7] = 0.33 * m;
     dXOut[7] = 0.77 * m;
     dYIn[7] = 0.85 * m;
     dYOut[7] = 2.41 * m;
     gapIn[7] = 0.09 * m;
     gapOut[7] = 0.26 * m;
     dZ[7] = dZ8 - zgap / 2;
     Z[7] = Z[6] + dZ[6] + dZ[7] + zgap;

     dXIn[8] = dXOut[7];
     dYIn[8] = dYOut[7];
     dXOut[8] = dXIn[8];
     dYOut[8] = dYIn[8];
     gapIn[8] = gapOut[7];
     gapOut[8] = gapIn[8];
     dZ[8] = 0.1 * m;
     Z[8] = Z[7] + dZ[7] + dZ[8];

     for (int i = 0; i < nMagnets; ++i) {
        midGapIn[i] = 0.;
        midGapOut[i] = 0.;
        HmainSideMagIn[i] = dYIn[i] / 2;
        HmainSideMagOut[i] = dYOut[i] / 2;
     }

  } else if (fDesign == 7) {
  magnetName = {"MagnAbsorb1", "MagnAbsorb2", "Magn1", "Magn2", "Magn3",
                "Magn4", "Magn5", "Magn6", "Magn7"};

  fieldDirection[0] = FieldDirection::up;
  dXIn[0]  = 0.4*m;			dYIn[0]	= 1.5*m;
  dXOut[0] = 0.40*m;			dYOut[0]= 1.5*m;
  gapIn[0] = 0.02 * m;			gapOut[0] = 0.02 * m;
  dZ[0] = dZ1-zgap/2;			Z[0] = zEndOfAbsorb + dZ[0]+zgap;

  fieldDirection[1] = FieldDirection::up;
  dXIn[1]  = 0.8*m;			dYIn[1]	= 1.5*m;
  dXOut[1] = 0.8*m;			dYOut[1]= 1.5*m;
  gapIn[1] = 0.02*m;			gapOut[1] = 0.02*m;
  dZ[1] = dZ2-zgap/2;			Z[1] = Z[0] + dZ[0] + dZ[1]+zgap;

  fieldDirection[2] = FieldDirection::up;
  dXIn[2]  = 0.87*m;			dYIn[2]	= 0.35*m;
  dXOut[2] = 0.65*m;			dYOut[2]= 1.21*m;
  gapIn[2] = 0.11 * m;  		gapOut[2] = 0.065 * m;
  dZ[2] = dZ3-zgap/2;			Z[2] = Z[1] + dZ[1] + dZ[2] + zgap;

  fieldDirection[3] = FieldDirection::up;
  dXIn[3]  = 0.65*m;			dYIn[3]	= 1.21*m;
  dXOut[3] = 0.43*m;			dYOut[3]= 2.07*m;
  gapIn[3] = 0.065 * m; 		gapOut[3] = 0.02 * m;
  dZ[3] = dZ4-zgap/2;			Z[3] = Z[2] + dZ[2] + dZ[3]+zgap;

  fieldDirection[4] = FieldDirection::up;
  dXIn[4]  = 0.06*m;			dYIn[4]	= 0.32*m;
  dXOut[4] = 0.33*m;			dYOut[4]= 0.13*m;
  gapIn[4] = 0.7*m;			gapOut[4] = 0.11*m;
  dZ[4] = dZ5-zgap/2;			Z[4] = Z[3] + dZ[3] + dZ[4]+zgap;

  fieldDirection[5] = FieldDirection::down;
  dXIn[5]  = 0.05*m;			dYIn[5]	= 1.12*m;
  dXOut[5] =0.16*m;			dYOut[5]= 0.05*m;
  gapIn[5] = 0.04*m;			gapOut[5] = 0.02*m;
  dZ[5] = dZ6-zgap/2;			Z[5] = Z[4] + dZ[4] + dZ[5]+zgap;

  fieldDirection[6] = FieldDirection::down;
  dXIn[6]  = 0.15*m;			dYIn[6]	= 2.35*m;
  dXOut[6] = 0.34*m;			dYOut[6]= 0.32*m;
  gapIn[6] = 0.05*m;			gapOut[6] = 0.08*m;
  dZ[6] = dZ7-zgap/2;			Z[6] = Z[5] + dZ[5] + dZ[6]+zgap;

  Double_t clip_width = 0.1*m; // clip field width by this width
  fieldDirection[7] = FieldDirection::down;
  dXIn[7]  = 0.31*m;			dYIn[7]	= 1.86*m;
  dXOut[7] = 0.9*m - clip_width;	dYOut[7]= 3.1*m;
  Double_t clip_len =
       (dZ8-zgap/2) * (1 - (dXOut[7] - dXIn[7]) / (dXOut[7] + clip_width - dXIn[7]));
  gapIn[7] = 0.02*m;			gapOut[7] = 0.55*m;
  dZ[7] = dZ8 - clip_len - zgap / 2;	Z[7] = Z[6] + dZ[6] + dZ[7] + zgap;

  fieldDirection[8] = FieldDirection::down;
  dXIn[8]  = dXOut[7];			dYIn[8]	= dYOut[7];
  dXOut[8] = dXOut[7];			dYOut[8]= dYOut[7];
  gapIn[8] = 0.55*m;			gapOut[8] = 0.55*m;
  dZ[8] = clip_len;			Z[8] = Z[7] + dZ[7] + dZ[8];

  for (int i = 0; i < nMagnets; ++i) {
    midGapIn[i] = 0.;
    midGapOut[i] = 0.;
    HmainSideMagIn[i] = dYIn[i] / 2;
    HmainSideMagOut[i] = dYOut[i] / 2;
  }

  } else {

  magnetName = {"1", "2", "3", "4", "5", "6", "7"};

  fieldDirection[0] = FieldDirection::up;
  dXIn[0]  = 0.7*m;			dYIn[0]	= 1.*m;
  dXOut[0] = 0.7*m;			dYOut[0]= 0.8158*m;
  midGapIn[0] = 0; 			midGapOut[0] = 0;
  HmainSideMagIn[0] = dYIn[0];  	HmainSideMagOut[0] = dYOut[0];
  gapIn[0] = 20;			gapOut[0] = 20;
  dZ[0] = dZ1-zgap;			Z[0] = zEndOfAbsorb + dZ[0]+zgap;

  fieldDirection[1] = FieldDirection::up;
  dXIn[1]  = 0.36*m;			dYIn[1]	= 0.8158*m;
  dXOut[1] = 0.19*m;			dYOut[1]= 0.499*m;
  midGapIn[1] = 0; 			midGapOut[1] = 0;
  HmainSideMagIn[1] = dYIn[1]/2;  	HmainSideMagOut[1] = dYOut[1]/2;
  gapIn[1] = 88;			gapOut[1] = 122;
  dZ[1] = dZ2-zgap/2;			Z[1] = Z[0] + dZ[0] + dZ[1]+zgap;

  fieldDirection[2] = FieldDirection::down;
  dXIn[2]  = 0.075*m;			dYIn[2]	= 0.499*m;
  dXOut[2] = 0.25*m;			dYOut[2]= 1.10162*m;
  midGapIn[2] = 0; 			midGapOut[2] = 0;
  HmainSideMagIn[2] = dYIn[2]/2;  	HmainSideMagOut[2] = dYOut[2]/2;
  gapIn[2] = 0;				gapOut[2] = 0;
  dZ[2] = dZ3-zgap/2;			Z[2] = Z[1] + dZ[1] + dZ[2]+zgap;

  fieldDirection[3] = FieldDirection::down;
  dXIn[3]  = 0.25*m;			dYIn[3]	= 1.10262*m;
  dXOut[3] = 0.3*m;			dYOut[3]= 1.82697*m;
  midGapIn[3] = 0; 			midGapOut[3] = 0;
  HmainSideMagIn[3] = dXIn[3];  	HmainSideMagOut[3] = dXOut[3];
  gapIn[3] = 0;				gapOut[3] = 25;
  dZ[3] = dZ4-zgap/2;			Z[3] = Z[2] + dZ[2] + dZ[3]+zgap;

  fieldDirection[4] = FieldDirection::down;
  dXIn[4]  = 0.3*m;			dYIn[4]	= 1.82697*m;
  dXOut[4] = 0.4*m;			dYOut[4]= 2.55131*m;
  midGapIn[4] = 5; 			midGapOut[4] = 25;
  HmainSideMagIn[4] = dXIn[4];  	HmainSideMagOut[4] = dXOut[4];
  gapIn[4] = 20;			gapOut[4] = 20;
  dZ[4] = dZ6-zgap/2;			Z[4] = Z[3] + dZ[3] + dZ[4]+zgap;

  fieldDirection[5] = FieldDirection::down;
  dXIn[5]  = 0.4*m;			dYIn[5]	= 2.55131*m;
  dXOut[5] =0.4*m;			dYOut[5]= 3.27566*m;
  midGapIn[5] = 25; 			midGapOut[5] = 65;
  HmainSideMagIn[5] = dXIn[5];  	HmainSideMagOut[5] = dXOut[5];
  gapIn[5] = 20;			gapOut[5] = 20;
  dZ[5] = dZ7-zgap/2;			Z[5] = Z[4] + dZ[4] + dZ[5]+zgap;

  fieldDirection[6] = FieldDirection::down;
  dXIn[6]  = 0.4*m;			dYIn[6]	= 3.27566*m;
  dXOut[6] = 0.75*m;			dYOut[6]= 4*m;
  midGapIn[6] = 65; 		        midGapOut[6] = 75;
  HmainSideMagIn[6] = dXIn[6];  	HmainSideMagOut[6] = dXOut[6];
  gapIn[6] = 20;			gapOut[6] = 20;
  dZ[6] = dZ8-zgap/2;			Z[6] = Z[5] + dZ[5] + dZ[6]+zgap;
  }
  return nMagnets;
}
void ShipMuonShield::ConstructGeometry()
{
    TGeoVolume *top=gGeoManager->GetTopVolume();
    TGeoVolume *tShield = new TGeoVolumeAssembly("MuonShieldArea");
    InitMedium("steel");
    TGeoMedium *steel =gGeoManager->GetMedium("steel");
    InitMedium("iron");
    TGeoMedium *iron  =gGeoManager->GetMedium("iron");
    InitMedium("Concrete");
    TGeoMedium *concrete  =gGeoManager->GetMedium("Concrete");

    if (fDesign >= 7 && fDesign <= 9) {
      Double_t ironField = fField*tesla;
      TGeoUniformMagField *magFieldIron = new TGeoUniformMagField(0.,ironField,0.);
      TGeoUniformMagField *RetField     = new TGeoUniformMagField(0.,-ironField,0.);
      TGeoUniformMagField *ConRField    = new TGeoUniformMagField(-ironField,0.,0.);
      TGeoUniformMagField *ConLField    = new TGeoUniformMagField(ironField,0.,0.);
      TGeoUniformMagField *fields[4] = {magFieldIron,RetField,ConRField,ConLField};

      std::vector<TString> magnetName;
      std::vector<FieldDirection> fieldDirection;
      std::vector<Double_t> dXIn, dYIn, dXOut, dYOut, dZf, midGapIn, midGapOut,
	  HmainSideMagIn, HmainSideMagOut, gapIn, gapOut, Z;
      const Int_t nMagnets = Initialize(magnetName, fieldDirection, dXIn, dYIn, dXOut, dYOut, dZf,
		 midGapIn, midGapOut, HmainSideMagIn, HmainSideMagOut, gapIn,
		 gapOut, Z);

      // Create TCC8 tunnel around muon shield
      Double_t TCC8_length =  170 * m;
      // Add small stair step at the beginning of ECN3
      Double_t stair_step_length = 0.82 * m;
      Double_t ECN3_length =  100 * m;
      Double_t TCC8_trench_length = 12 * m;
      Double_t zgap = 10 * cm;
      Double_t absorber_offset = zgap;
      Double_t absorber_half_length = (dZf[0] + dZf[1]) + zgap / 2.;
      Double_t z_transition = zEndOfAbsorb + 2 * absorber_half_length + absorber_offset + 14 * cm + TCC8_trench_length;
      auto *rock = new TGeoBBox("rock", 20 * m, 20 * m, TCC8_length / 2. + ECN3_length / 2. + 5 * m);
      auto *muon_shield_cavern = new TGeoBBox("muon_shield_cavern", 4.995 * m, 3.75 * m, TCC8_length / 2.);
      auto *TCC8_shift = new TGeoTranslation("TCC8_shift", 1.435 * m, 2.05 * m, - TCC8_length / 2.);
      TCC8_shift->RegisterYourself();

      // Create ECN3 cavern around vessel
      auto *experiment_rock = new TGeoBBox("experiment_rock", 20 * m, 20 * m, ECN3_length / 2.);
      auto *stair_step = new TGeoBBox("stair_step", 7.995 * m, 5.6 * m , stair_step_length / 2.);
      auto *stair_step_shift = new TGeoTranslation("stair_step_shift", 3.435 * m, 3.04 * m , stair_step_length / 2.);
      stair_step_shift->RegisterYourself();
      auto *experiment_cavern = new TGeoBBox("experiment_cavern", 7.995 * m, 6 * m, ECN3_length / 2. - stair_step_length / 2.);
      auto *ECN3_shift = new TGeoTranslation("ECN3_shift", 3.435 * m, 2.64 * m, ECN3_length / 2. + stair_step_length);
      ECN3_shift->RegisterYourself();

      auto *yoke_pit = new TGeoBBox("yoke_pit", 3.5 * m, 4.3 * m + 1 * cm, 2.5 * m);
      auto *yoke_pit_shift = new TGeoTranslation("yoke_pit_shift", 0 * m, 0 * m, 31 * m - z_transition);
      yoke_pit_shift->RegisterYourself();

      auto *target_pit = new TGeoBBox("target_pit", 2 * m, 0.5 * m, 2 * m);
      auto *target_pit_shift = new TGeoTranslation("target_pit_shift", 0 * m, -2.2 * m, zEndOfAbsorb - 2 * m - z_transition);
      target_pit_shift->RegisterYourself();

        float mField = 1.6 * tesla;
	TGeoUniformMagField *fieldsAbsorber[4] = {
	    new TGeoUniformMagField(0., mField, 0.),
	    new TGeoUniformMagField(0., -mField, 0.),
	    new TGeoUniformMagField(-mField, 0., 0.),
	    new TGeoUniformMagField(mField, 0., 0.)
	};

	for (Int_t nM = (fDesign == 7) ? 0 : 1; nM < 2; nM++) {
	  CreateMagnet(magnetName[nM], iron, tShield, fieldsAbsorber,
		       fieldDirection[nM], dXIn[nM], dYIn[nM], dXOut[nM],
		       dYOut[nM], dZf[nM], midGapIn[nM], midGapOut[nM],
		       HmainSideMagIn[nM], HmainSideMagOut[nM], gapIn[nM],
		       gapOut[nM], Z[nM], true, false);
	}

      std::vector<TGeoTranslation*> mag_trans;

      if (fDesign == 7) {
         auto mag1 = new TGeoTranslation("mag1", 0, 0, -dZ2);
         mag1->RegisterYourself();
	 mag_trans.push_back(mag1);
      }
      auto mag2 = new TGeoTranslation("mag2", 0, 0, +dZ1);
      mag2->RegisterYourself();
      mag_trans.push_back(mag2);

      auto abs = new TGeoBBox("absorber",  4.995 * m -0.001*m, 3.75 * m -0.001*m, absorber_half_length - 0.001);
      auto *absorber_shift = new TGeoTranslation("absorber_shift", 1.435 * m, 2.05 * m, 0);
      absorber_shift->RegisterYourself();

      const std::vector<TString> absorber_magnets =
         (fDesign == 7) ? std::vector<TString>{"MagnAbsorb1", "MagnAbsorb2"} : std::vector<TString>{"MagnAbsorb2"};
      const std::vector<TString> magnet_components = fDesign == 7 ? std::vector<TString>{
	  "_MiddleMagL", "_MiddleMagR",  "_MagRetL",    "_MagRetR",
	  "_MagCLB",     "_MagCLT",      "_MagCRT",     "_MagCRB",
	  "_MagTopLeft", "_MagTopRight", "_MagBotLeft", "_MagBotRight",
      }: std::vector<TString>{
	  "_MiddleMagL", "_MiddleMagR",  "_MagRetL",    "_MagRetR",
	  "_MagTopLeft", "_MagTopRight", "_MagBotLeft", "_MagBotRight",
      };
      TString absorber_magnet_components;
      for (auto &&magnet_component : magnet_components) {
	// format: "-<magnetName>_<magnet_component>:<translation>"
	absorber_magnet_components +=
	    ("-" + absorber_magnets[0] + magnet_component + ":" +
	     mag_trans[0]->GetName());
	if (fDesign == 7) {
	absorber_magnet_components +=
	    ("-" + absorber_magnets[1] + magnet_component + ":" +
	     mag_trans[1]->GetName());
	}
      }
      TGeoCompositeShape *absorberShape = new TGeoCompositeShape(
	  "Absorber", "absorber:absorber_shift" + absorber_magnet_components); // cutting out
								// magnet parts
								// from absorber
      TGeoVolume *absorber = new TGeoVolume("AbsorberVol", absorberShape, iron);
      absorber->SetLineColor(42); // brown / light red
      tShield->AddNode(absorber, 1, new TGeoTranslation(0, 0, zEndOfAbsorb + absorber_half_length + absorber_offset));

      auto *compRock = new TGeoCompositeShape("compRock",
                                              "rock - muon_shield_cavern:TCC8_shift"
                                              "- experiment_cavern:ECN3_shift"
                                              "- stair_step:stair_step_shift"
                                              "- yoke_pit:yoke_pit_shift"
                                              "- target_pit:target_pit_shift"
      );
      auto *Cavern = new TGeoVolume("Cavern", compRock, concrete);
      Cavern->SetLineColor(11);  // grey
      Cavern->SetTransparency(50);
      top->AddNode(Cavern, 1, new TGeoTranslation(0, 0, z_transition));


      std::array<double, 9> fieldScale = {{1., 1., 1., 1., 1., 1., 1., 1., 1.}};
      if (fWithCoMagnet > 0)
      {
        Double_t lengthSum = 0.;
        for (int i = 2; i < 9; ++i)
        {
          lengthSum += dZf[i];
        }
        fieldScale.fill((fField * lengthSum -  2.2 * dZf[fWithCoMagnet])/fField/(lengthSum - dZf[fWithCoMagnet]));
        fieldScale[0] = 1.;
        fieldScale[1] = 1.;
        try
        {
         fieldScale.at(fWithCoMagnet) = 2.2 / fField;
        }
        catch(const std::out_of_range& e)
        {
           Fatal( "ShipMuonShield", "Exception out of range for --coMuonShield occurred \n");
        }
      }
      for (Int_t nM = 2; nM <= (nMagnets - 1); nM++) {

          // SC MAGNET
  if ((dZf[nM] < 1e-5 || nM == 4 ) && fSC_mag) {
        continue;
      }
  Double_t ironField_s_SC = fField * fieldScale[nM] * tesla;
  Bool_t SC_key = false;
  if(nM == 3 && fSC_mag){
    Double_t SC_FIELD = 5.1;
    ironField_s_SC = SC_FIELD * fieldScale[nM] * tesla;
    SC_key = true;
  }
  // END
  Double_t ironField_s = fField * fieldScale[nM] * tesla;
  TGeoUniformMagField *magFieldIron_s = new TGeoUniformMagField(0.,ironField_s_SC,0.);
  TGeoUniformMagField *RetField_s     = new TGeoUniformMagField(0.,-ironField_s,0.);
  TGeoUniformMagField *ConRField_s    = new TGeoUniformMagField(-ironField_s,0.,0.);
  TGeoUniformMagField *ConLField_s    = new TGeoUniformMagField(ironField_s,0.,0.);
  TGeoUniformMagField *fields_s[4] = {magFieldIron_s,RetField_s,ConRField_s,ConLField_s};
  Int_t split_num = 20;
  if(nM >= 4){
    Double_t spl_shift_X = (dXOut[nM] - dXIn[nM]) / split_num;
    Double_t spl_shift_Y = (dYOut[nM] - dYIn[nM]) / split_num;
    Double_t spl_shift_midgap = (midGapOut[nM] - midGapIn[nM]) / split_num;
    Double_t spl_shift_gap = (gapOut[nM] - gapIn[nM]) / split_num;
    Double_t spl_shift_HmainsideMag = (HmainSideMagOut[nM] - HmainSideMagIn[nM]) / split_num;
    Double_t spl_dZ = dZf[nM] / split_num;
    // Double_t Z_shift = Z[nM];
    for(int spl = 0; spl < split_num; spl++){
      Double_t dXIn_spl = dXIn[nM] + spl_shift_X*spl;
      // Double_t dXIn_spl = dXIn[nM];
      Double_t dXOut_spl = dXIn[nM] + spl_shift_X*(spl+1);
      Double_t dYIn_spl = dYIn[nM] + spl_shift_Y*spl;
      // Double_t dYIn_spl = dYIn[nM];
      Double_t dYOut_spl = dYIn[nM] + spl_shift_Y*(spl+1);  
      Double_t midgapIn_spl = midGapIn[nM] + spl_shift_midgap*spl;
      // Double_t midgapIn_spl = midGapIn[nM];
      Double_t midgapOut_spl = midGapIn[nM] + spl_shift_midgap*(spl+1);    
      Double_t gapIn_spl = gapIn[nM] + spl_shift_gap*spl;
      // Double_t gapIn_spl = gapIn[nM];
      Double_t gapOut_spl = gapIn[nM] + spl_shift_gap*(spl+1);
      Double_t HmainSideMagIn_spl = HmainSideMagIn[nM] + spl_shift_HmainsideMag*spl;
      // Double_t HmainSideMagIn_spl = HmainSideMagIn[nM];
      Double_t HmainSideMagOut_spl = HmainSideMagIn[nM] + spl_shift_HmainsideMag*(spl+1);
      Double_t gap_btw = 0.15;
      // if(spl == 0) gap_btw = 0.0;
      CreateMagnet(magnetName[nM] + TString::Format("_%d", spl), iron, tShield, fields_s, fieldDirection[nM],
            dXIn_spl, dYIn_spl, dXOut_spl, dYOut_spl, spl_dZ - gap_btw,
           midgapIn_spl, midgapOut_spl, HmainSideMagIn_spl,
            HmainSideMagOut_spl, gapIn_spl, gapOut_spl, Z[nM] + (2*spl - (split_num-1))*spl_dZ, nM==8, fStepGeo, SC_key);
    }
  }
  else{
    CreateMagnet(magnetName[nM], iron, tShield, fields_s, fieldDirection[nM],
          dXIn[nM], dYIn[nM], dXOut[nM], dYOut[nM], dZf[nM],
          midGapIn[nM], midGapOut[nM], HmainSideMagIn[nM],
          HmainSideMagOut[nM], gapIn[nM], gapOut[nM], Z[nM], nM==8, fStepGeo, SC_key);
  }

	if (nM==8 || !fSupport) continue;
	Double_t dymax = std::max(dYIn[nM] + dXIn[nM], dYOut[nM] + dXOut[nM]);
	Double_t dymin = std::min(dYIn[nM] + dXIn[nM], dYOut[nM] + dXOut[nM]);
	Double_t slope =
	    (dYIn[nM] + dXIn[nM] - dYOut[nM] - dXOut[nM]) / (2 * dZf[nM]);
	Double_t w1 = 2 * dXIn[nM] + std::max(20., gapIn[nM]);
	Double_t w2 = 2 * dXOut[nM] + std::max(20., gapOut[nM]);
	Double_t anti_overlap = 0.1;
	Double_t h1 = 0.5 * (dYIn[nM] + dXIn[nM] + anti_overlap - 10 * m + fFloor);
	Double_t h2 = 0.5 * (dYOut[nM] + dXOut[nM] + anti_overlap - 10 * m + fFloor);
	Double_t length = std::min(0.5 * m, std::abs(dZf[nM]/2. - 5 * cm));
	std::array<Double_t, 16> verticesIn = {
	    -w1, -h1,
	    +w1, -h1,
	    +w1, +h1,
	    -w1, +h1,
	    -w1, -h1 + slope * 2. * length,
	    +w1, -h1 + slope * 2. * length,
	    +w1, +h1,
	    -w1, +h1,
	};
	std::array<Double_t, 16> verticesOut = {
	    -w2, -h2 - slope * 2. * length,
	    +w2, -h2 - slope * 2. * length,
	    +w2, +h2,
	    -w2, +h2,
	    -w2, -h2,
	    +w2, -h2,
	    +w2, +h2,
	    -w2, +h2,
	};
  if (!fStepGeo)
  {


  	TGeoVolume *pillar1 =
  	    gGeoManager->MakeArb8(TString::Format("pillar_%d", 2 * nM - 1),
  				  steel, length, verticesIn.data());
  	TGeoVolume *pillar2 =
  	    gGeoManager->MakeArb8(TString::Format("pillar_%d", 2 * nM), steel,
  				  length, verticesOut.data());
  	pillar1->SetLineColor(kGreen-5);
  	pillar2->SetLineColor(kGreen-5);
  	tShield->AddNode(pillar1, 1, new TGeoTranslation(
  				     0, -0.5 * (dYIn[nM] + dXIn[nM] + 10 * m - fFloor),
  				     Z[nM] - dZf[nM] + length));
  	tShield->AddNode(pillar2, 1, new TGeoTranslation(
  				     0, -0.5 * (dYOut[nM] + dXOut[nM] + 10 * m - fFloor),
  				     Z[nM] + dZf[nM] - length));
  }
      }

      // Place in origin of SHiP coordinate system as subnodes placed correctly
      top->AddNode(tShield, 1);

    } else {
     Fatal("ShipMuonShield","Design %i does not match implemented designs",fDesign);
    }
}
