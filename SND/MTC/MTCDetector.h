// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: Copyright CERN for the benefit of the SHiP
// Collaboration

#ifndef SND_MTC_MTCDETECTOR_H_
#define SND_MTC_MTCDETECTOR_H_

#include <array>
#include <map>
#include <string>  // for string
#include <vector>

#include "Detector.h"
#include "MTCDetPoint.h"
#include "TGeoMatrix.h"

class TGeoVolume;
class TGeoVolumeAssembly;
class TGeoMedium;
class FairVolume;

class MTCDetector : public SHiP::Detector<MTCDetPoint> {
 public:
  MTCDetector(const char* name, Bool_t Active, const char* Title = "",
              Int_t DetId = 0);
  MTCDetector();

  void SetMTCParameters(Double_t width, Double_t height,
                        Double_t fiber_tilt_angle, Double_t iron_thickness,
                        Double_t scifi_thickness, Int_t num_of_agg_channels,
                        Double_t scint_cell_size, Double_t scint_thickness,
                        Int_t number_of_layers, Double_t z_position,
                        Double_t field_strength);
  virtual void CreateScintModule(const char* name,
                                 TGeoVolumeAssembly* modMotherVol,
                                 Double_t z_shift, Double_t width,
                                 Double_t height, Double_t thickness,
                                 Double_t cellSizeX, Double_t cellSizeY,
                                 TGeoMedium* material, Int_t color,
                                 Double_t transparency, Int_t LayerId);
  virtual void CreateSciFiModule(const char* name,
                                 TGeoVolumeAssembly* modMotherVol,
                                 Double_t width, Double_t height,
                                 Double_t thickness, Int_t iBlock);
  void ConstructGeometry() override;
  /** Get position of single fibre in global coordinate system**/
  void GetPosition(Int_t fDetectorID, TVector3& vLeft,
                   TVector3& vRight);  // or top and bottom
  /** Transform global position to local position in plane **/
  TVector3 GetLocalPos(Int_t fDetectorID, TVector3* glob);
  /** mean position of fibre2 associated with SiPM channel **/
  void GetSiPMPosition(Int_t SiPMChan, TVector3& A, TVector3& B);
  void SiPMmapping();
  std::map<Int_t, std::map<Int_t, std::array<float, 2>>> GetSiPMmapU(Int_t iB) {
    return fibresSiPM_U.at(iB);
  }
  std::map<Int_t, std::map<Int_t, std::array<float, 2>>> GetFibresMapU(Int_t iB) {
    return siPMFibres_U.at(iB);
  }
  std::map<Int_t, std::map<Int_t, std::array<float, 2>>> GetSiPMmapV(Int_t iB) {
    return fibresSiPM_V.at(iB);
  }
  std::map<Int_t, std::map<Int_t, std::array<float, 2>>> GetFibresMapV(Int_t iB) {
    return siPMFibres_V.at(iB);
  }
  std::map<Int_t, float> GetSiPMPos_U(Int_t iB) { return SiPMPos_U.at(iB); }
  std::map<Int_t, float> GetSiPMPos_V(Int_t iB) { return SiPMPos_V.at(iB); }
  Int_t Get_NSiPMChan(Int_t iB) const { return fNSiPMChan.at(iB); }
  Float_t Get_SciFiActiveX() const { return fSciFiActiveX; }
  /** Number of longitudinal MTC blocks (segments) **/
  Int_t GetNBlocks() const { return fnB; }
  /** Number of sandwich layers per block **/
  Int_t GetNLayersPerBlock() const { return fnLayPerBlock; }
  /** Map a global layer index (0..fLayers-1) to its block index (0..fnB-1) **/
  Int_t BlockForLayer(Int_t layer) const {
    Int_t b = (fnLayPerBlock > 0) ? layer / fnLayPerBlock : 0;
    return (b < fnB) ? b : fnB - 1;
  }
  virtual void SiPMOverlap(Int_t iB);
  Bool_t ProcessHits(FairVolume* vol = nullptr) override;

 private:
  Double_t fWidth;
  Double_t fHeight;
  Double_t fSciFiActiveX;
  Double_t fSciFiActiveY;
  Double_t fSciFiBendingAngle;
  Double_t fIronThick;
  Double_t fSciFiThick;
  Double_t fScintThick;
  Double_t fScintCellSize;
  Int_t fLayers;
  Double_t fZCenter;
  Double_t fFieldY;
  Double_t fZEpoxyMat;
  Double_t fiberMatThick = 0.135;      // 1.35 mm
  std::vector<Double_t> fFiberLength;  //! per-block fibre length
  Double_t fFiberPitch = 0.025;        // cm
  Int_t fnB;                           // number of longitudinal blocks
  Int_t fnLayPerBlock;                 // sandwich layers per block
  std::vector<Double_t> fBlockWidth;   //! per-block transverse width
  std::vector<Double_t> fBlockHeight;  //! per-block transverse height
                                 // Define sublayer thicknesses (in cm)
  // These values mimic the GEANT4 setup:
  Double_t lowerIronThick = 0.3;  // 3 mm
  Double_t airGap = 0.1;          // 1 mm
  Double_t upperIronThick = 0.3;  // 3 mm
  Double_t zLowerIronInt = -3.5 / 10;
  Double_t zFiberMat1 = -1.325 / 10;
  Double_t zAirGap = -0.15 / 10;
  Double_t zFiberMat2 = 1.025 / 10;
  Double_t zUpperIronInt = 3.2 / 10;
  Double_t fFiberRadius = 0.01125;
  Int_t numFiberLayers = 6;        // number of fiber layers in epoxy block
  std::vector<Int_t> fNSiPMChan;   //! per-block number of SiPM channels
  Int_t fChannelAggregated;        // Number of SiPM channels to be aggregated
  std::vector<Int_t> fNSiPMs;      //! per-block number of SiPMs
  static constexpr Int_t kMaxChannelsPerSiPM = 1000;
  // Total module thickness = 0.3 + 0.135 + 0.1 + 0.135 + 0.3 ≈ 1.0 cm
  Int_t fNMats = 1;
  std::vector<std::map<Int_t, std::map<Int_t, std::array<float, 2>>>>
      fibresSiPM_U;  //! per-block mapping of fibres to SiPM channels
  std::vector<std::map<Int_t, std::map<Int_t, std::array<float, 2>>>>
      siPMFibres_U;  //! per-block inverse mapping
  std::vector<std::map<Int_t, std::map<Int_t, std::array<float, 2>>>>
      fibresSiPM_V;  //! per-block mapping of fibres to SiPM channels
  std::vector<std::map<Int_t, std::map<Int_t, std::array<float, 2>>>>
      siPMFibres_V;  //! per-block inverse mapping
  std::vector<std::map<Int_t, float>>
      SiPMPos_U, SiPMPos_V;  //! per-block local SiPM channel position

  MTCDetector(const MTCDetector&) = delete;
  MTCDetector& operator=(const MTCDetector&) = delete;
  ClassDefOverride(MTCDetector, 5)
};

#endif  // SND_MTC_MTCDETECTOR_H_
