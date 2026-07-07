#ifndef SND_SIWCALO_SIWCALOHIT_H_
#define SND_SIWCALO_SIWCALOHIT_H_ 1

#include "ShipHit.h"
#include "SiFeCaloPoint.h"
#include "TObject.h"
#include "TVector3.h"

class SiFeCaloHit : public ShipHit
{
  public:
    /** Default constructor **/
    SiFeCaloHit();
    // SiFeCaloHit(Int_t detID);

    //  Constructor from SiFeCaloPoint
    SiFeCaloHit(Int_t detID, const std::vector<SiFeCaloPoint*>&);

    /** Destructor **/
    virtual ~SiFeCaloHit();

    /** Output to screen **/
    void Print();
    // void Print() const;
    Float_t GetSignal() { return fSignal; };
    Double_t GetX() { return fX; }
    Double_t GetY() { return fY; }
    Double_t GetZ() { return fZ; }
    
    int constexpr GetLayer() { return fDetectorID >> 17;}
    int constexpr GetColumn() { return (fDetectorID >> 14) & 0x3;}
    int constexpr GetRow() { return (fDetectorID >> 13) & 0x1;}
    int constexpr GetPixelX() { return fDetectorID & 0x1F;} // 0-31
    int constexpr GetPixelY() { return (fDetectorID >> 5) & 0x1F;} // 0-31  
    // Helpers                                                           
    int constexpr GetPixelID() { return GetPixelX() + 32 * GetPixelY();} // 0-1023
  
    bool isValid() const { return flag; }

  private:
    /** Copy constructor **/
    SiFeCaloHit(const SiFeCaloHit& hit) = default;
    SiFeCaloHit& operator=(const SiFeCaloHit& hit) = default;

    Float_t fSignal;
    Double_t fX;
    Double_t fY;
    Double_t fZ;
    Float_t flag;

    ClassDef(SiFeCaloHit, 1);
};

#endif   // SND_SIWCALO_SIWCALOHIT_H_
