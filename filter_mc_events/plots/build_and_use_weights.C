/*******************************************************
 *  build_and_use_weights.C
 *
 *  root -l -q build_and_use_weights.C'()'
 ******************************************************/
#include <ROOT/RDataFrame.hxx>
#include <TH1F.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TLegend.h>

// ------------------------------------------------------------------
// user configuration in ONE place
// ------------------------------------------------------------------
namespace cfg {
  constexpr const char* file2x2   = "../track_data.root";
  constexpr const char* filetred  = "../lifetime1ms/many_muon_hits.root";
  constexpr const char* tree2x2   = "selected_data/hits";
  constexpr const char* treetred  = "mu_ndlar/hits";
  constexpr int    nThresBins     = 64;      // change if you want
  constexpr float  thresMin       = -0.5;
  constexpr float  thresMax       = 63.5;
}

// ------------------------------------------------------------------
// 1) build weight histogram and store it on disk
// ------------------------------------------------------------------
TH1F* BuildWeightHist()
{
  using namespace cfg;

  ROOT::RDataFrame d2x2 (tree2x2 , file2x2 );
  ROOT::RDataFrame dtred(treetred, filetred);

  auto h2x2  = d2x2 .Filter("tindex==0")
                     .Histo1D({"h2x2_thres", "", nThresBins, thresMin, thresMax},
                               "thres");
  auto hTred = dtred.Filter("tindex==0")
                     .Histo1D({"hTred_thres","", nThresBins, thresMin, thresMax},
                               "thres");

  h2x2->Scale(1./h2x2->Integral());
  hTred->Scale(1./hTred->Integral());

  auto* hW = static_cast<TH1F*>(h2x2->Clone("hThresWeight"));
  hW->Divide(hTred.GetPtr());
  hW->SetDirectory(0);

  TFile fout("weights.root","RECREATE");
  hW->Write();
  fout.Close();

  return hW;          // pointer lives after file is closed (SetDirectory(0))
}

// ------------------------------------------------------------------
// 2) helper that any RDataFrame can use
// ------------------------------------------------------------------
TH1F* gThresW = nullptr;

float GetThresWeight(double thr)
{
  if (!gThresW)
    throw std::runtime_error("Weight histogram not initialised!");
  int bin = gThresW->FindFixBin(thr);
  return gThresW->GetBinContent(bin);
}

// ------------------------------------------------------------------
// 3) example: compare totQ with thres-re-weighted tred
// ------------------------------------------------------------------
void Compare_totQ_Weighted()
{
  using namespace cfg;

  // make or read the weight histogram -------------------------------
  std::unique_ptr<TFile> fW(TFile::Open("weights.root"));
  if (fW && !fW->IsZombie())
    gThresW = static_cast<TH1F*>(fW->Get("hThresWeight"));
  if (!gThresW)                    // build if file not present
    gThresW = BuildWeightHist();

  //------------------------------------------------------------------
  // build histograms
  //------------------------------------------------------------------
  ROOT::RDataFrame d2x2 (tree2x2 , file2x2 );
  ROOT::RDataFrame dtred(treetred, filetred);

  auto h2x2 = d2x2.Filter("tindex==0")
                   .Histo1D({"h2x2_totQ","",45,0,45},"totQ");

  auto dtredW = dtred.Filter("tindex==0")
                      .Define("w",&GetThresWeight,{"thres"});

  auto hTredW = dtredW.Histo1D({"hTred_totQ_w","",45,0,45},
                               "totQ","w");
  auto hTred = dtredW.Histo1D({"hTred_totQ","",45,0,45},
                               "totQ");

  //------------------------------------------------------------------
  // normalise & draw
  //------------------------------------------------------------------
  h2x2  ->Scale(1./h2x2 ->Integral());
  hTredW->Scale(1./hTredW->Integral());
  hTred->Scale(1./hTred->Integral());

  TCanvas c("c_totQ_w","totQ weighted",800,600);
  h2x2 ->SetLineColor(kRed);
  hTredW->SetLineColor(kBlue);
  hTred->SetLineColor(kGreen);

  h2x2->SetTitle("totQ   (tindex==0)   with thres weights;totQ;normalised counts");
  h2x2 ->Draw();
  hTredW->Draw("SAME");
  hTred->Draw("SAME");

  TLegend leg(0.55,0.7,0.85,0.85);
  leg.AddEntry(h2x2 .GetPtr(),"2x2 (reference)","l");
  leg.AddEntry(hTredW.GetPtr(),"tred (weighted)","l");
  leg.AddEntry(hTred.GetPtr(),"tred (no weight)","l");
  leg.Draw();

  c.Print("comp_totQ_weighted.png");
}

void Compare_totN_Weighted()
{
  using namespace cfg;

  // make or read the weight histogram -------------------------------
  std::unique_ptr<TFile> fW(TFile::Open("weights.root"));
  if (fW && !fW->IsZombie())
    gThresW = static_cast<TH1F*>(fW->Get("hThresWeight"));
  if (!gThresW)                    // build if file not present
    gThresW = BuildWeightHist();

  //------------------------------------------------------------------
  // build histograms
  //------------------------------------------------------------------
  ROOT::RDataFrame d2x2 (tree2x2 , file2x2 );
  ROOT::RDataFrame dtred(treetred, filetred);

  auto h2x2 = d2x2.Filter("tindex==0")
                   .Histo1D({"h2x2_totN","",5,-0.5,4.5},"totN");

  auto dtredW = dtred.Filter("tindex==0")
                      .Define("w",&GetThresWeight,{"thres"});

  auto hTredW = dtredW.Histo1D({"hTred_totN_w","",5,-0.5,4.5},
                               "totN","w");
  auto hTred = dtredW.Histo1D({"hTred_totN","",5,-0.5,4.5},
                               "totN");

  //------------------------------------------------------------------
  // normalise & draw
  //------------------------------------------------------------------
  h2x2  ->Scale(1./h2x2 ->Integral());
  hTredW->Scale(1./hTredW->Integral());
  hTred->Scale(1./hTred->Integral());

  TCanvas c("c_totN_w","totN weighted",800,600);
  h2x2 ->SetLineColor(kRed);
  hTredW->SetLineColor(kBlue);
  hTred->SetLineColor(kGreen);

  h2x2->SetTitle("totN   (tindex==0)   with thres weights;totN;normalised counts");
  h2x2 ->Draw();
  hTredW->Draw("SAME");
  hTred->Draw("SAME");

  TLegend leg(0.55,0.7,0.85,0.85);
  leg.AddEntry(h2x2 .GetPtr(),"2x2 (reference)","l");
  leg.AddEntry(hTredW.GetPtr(),"tred (weighted)","l");
  leg.AddEntry(hTred.GetPtr(),"tred (no weight)","l");
  leg.Draw();

  c.Print("comp_totN_weighted.png");
}

// ------------------------------------------------------------------
// master function that users call
// ------------------------------------------------------------------
void build_and_use_weights()
{
    BuildWeightHist();
  // Build weight table (if not already on disk) and run one example.
  // Extend with further comparisons exactly like Compare_totQ_Weighted().
  Compare_totQ_Weighted();
  Compare_totN_Weighted();
}

