#ifndef MtcDetHit_H_
#define MtcDetHit_H_ 1


#include "ShipHit.h"
#include "MtcDetPoint.h"
#include "TObject.h"
#include "TVector3.h"


class MtcDetPoint;
class FairVolume;

class MtcDetHit : public ShipHit
{
  public:

    /** Default constructor **/
    MtcDetHit();
    /** Copy constructor **/
    MtcDetHit(const MtcDetHit& hit) = default;
    MtcDetHit& operator=(const MtcDetHit& hit) = default;
    //  Constructor from MtcDetPoint
    MtcDetHit(int detID,std::vector<MtcDetPoint*>,std::vector<Float_t>);

 /** Destructor **/
    virtual ~MtcDetHit();

    /** Output to screen **/
	void Print() ;
	Float_t GetSignal();
  Float_t GetTime();
  Float_t GetEnergy();
	void setInvalid() {flag = false;}
	bool isValid() const {return flag;}
  /*
    Example of fiberID: 123051820, where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees, 2 for scint plane)
      - 5: z-layer number (0-5)
      - 1820: local fibre ID within the station
    Example of SiPM global channel (what is seen in the output file): 123004123, where:
      - 1: MTC unique ID
      - 23: layer number
      - 0: station type (0 for +5 degrees, 1 for -5 degrees)
      - 0: mat number (only 0 by June 2025)
      - 4: SiPM number (0-N, where N is the number of SiPMs in the station)
      - 123: number of the SiPM channel (0-127, 128 channels per SiPM)
  */
	Int_t GetStation(){return floor(fDetectorID/1000000);}
	bool isVertical(){
		if ( int(fDetectorID/100000)%10 == 1){return true;}
		else{return (false);}
  }
/*
	SND@LHC comment: from Guido (22.9.2021): A threshold of 3.5pe should be used, which corresponds to 0.031MeV.
	1 SiPM channel has 104 pixels, pixel can only see 0 or >0 photons.
*/
  private:
    Float_t signals = 0;
    Float_t time;
    Float_t ly_loss(Float_t distance);
    Float_t sipm_saturation(Float_t ly, Float_t nphe_max);
    Float_t npix_to_qdc(Float_t npix);
    Float_t flag;   ///< flag

    ClassDef(MtcDetHit, 4);

};

#endif