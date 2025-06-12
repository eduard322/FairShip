#include "MtcDetHit.h"
#include "MtcDetPoint.h"
#include "MTCDetector.h"
#include "TROOT.h"
#include "FairRunSim.h"
#include "TGeoNavigator.h"
#include "TGeoManager.h"
#include "TGeoBBox.h"
#include <TRandom.h>
#include <iomanip>

namespace {
     //parameters for simulating the digitized information
     const Float_t ly_loss_params[4] = {20., 300.}; //x_0, lambda
     const Float_t npix_to_qdc_params[4] = {0.172, -1.31, 0.006, 0.33}; // A, B, sigma_A, sigma_B
}

// -----   Default constructor   -------------------------------------------
MtcDetHit::MtcDetHit()
  : ShipHit()
{
 flag = true;
}


// -----   constructor from point class  ------------------------------------------
MtcDetHit::MtcDetHit (int SiPMChan, std::vector<MtcDetPoint*> V, std::vector<Float_t> W)
{
     MTCDetector* MTCDet = dynamic_cast<MTCDetector*> (gROOT->GetListOfGlobals()->FindObject("MTC") );
     Float_t nphe_min = 3.5;
     Float_t nphe_max = 104.;
     Float_t timeResol = 150.*0.001; // in picoseconds
     Float_t signalSpeed = 15; // in cm/ns, speed of light in scintillating fibers
    //  fDetectorID  = SiPMChan;
     Float_t ly_total = 0;
     Float_t earliestToA   = 1E20;
     Int_t plane_type = int(SiPMChan/100000)%10; // 0 - U, 1 - V
     for( int i = 0; i <V.size();i++) {

        if(plane_type == 2){
          signals += V[i]->GetEnergyLoss(); // signal from Scintillating mat
          std::cout << Form("MtcDetHit: Scintillating mat, SiPM %d, signal %.2f keV, particle: %d", SiPMChan, signals*1e6, V[i]->PdgCode()) << std::endl;
        }
        else{

          Double_t signal = V[i]->GetEnergyLoss()*W[i];
    // Find distances from MCPoint centre to ends of fibre
          TVector3 a, b;
          TVector3 impact(V[i]->GetX(),V[i]->GetY() ,V[i]->GetZ() );
          MTCDet->GetSiPMPosition(SiPMChan, a, b);
          Double_t distance;
          // Calculate distance from energy deposit to SiPM.
          distance = (b - impact).Mag();
          // convert energy deposit to light yield (here Np.e. == avg. N fired pixels)
          Float_t ly = signal*1E+6*0.16; //0.16 p.e per 1 keV
          // account for the light attenuation in the fibers
          std::cout << Form("MtcDetHit: SiPM %d, Fibre: %d, distance %.2f cm, signal %.2f keV, light yield %.2f p.e., particle: %d", SiPMChan, V[i]->GetDetectorID(), distance, signal*1e6, ly, V[i]->PdgCode()) << std::endl;
          ly*= ly_loss(distance);
          ly_total+= ly;

          // for the timing, find earliest light to arrive at SiPM and smear with time resolution
          Float_t arrival_time = V[i]->GetTime() + distance/signalSpeed;
          if (arrival_time < earliestToA){earliestToA = arrival_time;}
          }
    }
     time = gRandom->Gaus(earliestToA, timeResol);
     if(plane_type == 2){
          flag=true;
          std::cout << Form("MtcDetHit: Scintillating mat, SiPM %d, total signal %.2f MeV", SiPMChan, signals) << std::endl;
     }
     else{
          // smear the total light yield using Poisson distribution
          ly_total = gRandom->Poisson(ly_total);
          // account for limited SiPM dyn. range
          Float_t Npix = sipm_saturation(ly_total, nphe_max);
          // convert Npix to QDC
          signals = npix_to_qdc(Npix);
          if (ly_total > nphe_min){   // nominal threshold at 3.5 p.e.
            flag=true;
          }else{
            flag=false;
          }
          std::cout << Form("MtcDetHit: SiPM %d, total light yield %.2f p.e., Npix %.2f, signal %.2f, time %.3f ns \n", SiPMChan, ly_total, Npix, signals, time);
     }
}

// -----   Destructor   ----------------------------------------------------
MtcDetHit::~MtcDetHit() { }
// -------------------------------------------------------------------------

// -----   Public method GetEnergy   -------------------------------------------
Float_t MtcDetHit::GetEnergy()
{
  // to be calculated from digis and calibration constants, missing!
  return signals;
}

Float_t MtcDetHit::ly_loss(Float_t distance){
//	It returns the light yield attenuation depending on the distance to SiPM
	return TMath::Exp(-(distance-ly_loss_params[0])/ly_loss_params[1]);
}

Float_t MtcDetHit::sipm_saturation(Float_t ly, Float_t nphe_max){
//	It returns the number of fired pixels per channel
        Float_t factor = 1 - TMath::Exp(-ly/nphe_max);
	return nphe_max*factor;
}

Float_t MtcDetHit::npix_to_qdc(Float_t npix){
//	It returns QDC per channel after Gaussian smearing of the parameters
        Float_t A = gRandom->Gaus(npix_to_qdc_params[0], npix_to_qdc_params[2]);
        Float_t B = gRandom->Gaus(npix_to_qdc_params[1], npix_to_qdc_params[3]);
        return A*npix + B;
}

// -----   Public method Print   -------------------------------------------
void MtcDetHit::Print()
{
  std::cout << "-I- MtcDetHit: Scifi hit " << " in station " << std::endl;
}
// -------------------------------------------------------------------------

ClassImp(MtcDetHit)
