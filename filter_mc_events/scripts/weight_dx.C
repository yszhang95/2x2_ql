#include <iostream>
#include "ROOT/RDataFrame.hxx"
#include "TH1F.h"
#include <map>

void weight_dx(int totN_min, int totN_max)
{
  using namespace ROOT;
 auto dfdata = RDataFrame("selected_data/hits", "merged_data.root");
 auto dftred = RDataFrame("selected_data/hits", "merged_hits.root");
 auto hdata = dfdata.Histo1D({"h_totN_data", "", 4, 0.5, 4.5}, "totN");
 auto htred = dftred.Histo1D({"h_totN_tred", "", 4, 0.5, 4.5}, "totN");

 auto weight = (TH1F*)hdata->Clone();
 weight->Divide(htred.GetPtr());
 weight->SetBinContent(0, 0);
 weight->SetBinContent(5, 0);
for(int i=1; i<=weight->GetNbinsX(); i++){
std::cout << "bin " << i << ", data: " << hdata->GetBinContent(i) << ", tred: " << htred->GetBinContent(i) << ", weight: " << weight->GetBinContent(i) << std::endl;
}

auto dfdata_weight = dfdata.Define("weight", [=](const long long totN) -> float { if (totN <=totN_max && totN >=totN_min) return 1.; else return 0.;}, {"totN"});
auto dftred_weight = dftred.Define("weight", [=,&weight](const long long totN) -> float { if (totN <=totN_max && totN >=totN_min) {return weight->GetBinContent(weight->FindBin(totN));} else {return 0.;}}, {"totN"}).
Define("totN123",  [=](const long long totN) -> float { if (totN <=totN_max && totN >=totN_min) return 1.; else return 0.;}, {"totN"});

  auto hdata_dx_weight = dfdata_weight.Histo1D({"hdata_dx_weight",
                                                ::Form("hdata_dx totN [%d, %d]", totN_min, totN_max), 50, -2, 3}, "dx", "weight");
  auto htred_dx_weight = dftred_weight.Histo1D({"htred_dx_weight",
                                                ::Form("htred_dx, totN [%d, %d], weighted", totN_min, totN_max), 50, -2, 3}, "dx", "weight");
  // auto hdata_dx = dfdata.Histo1D({"hdata_dx", "hdata_dx", 50, -2, 3}, "dx");
  auto htred_dx = dftred_weight.Histo1D({"htred_dx", ::Form("htred_dx, totN [%d, %d]", totN_min, totN_max), 50, -2, 3}, "dx", "totN123");

  hdata_dx_weight->Scale(1./hdata_dx_weight->Integral());
  htred_dx_weight->Scale(1./htred_dx_weight->Integral());
  htred_dx->Scale(1./htred_dx->Integral());

  TCanvas* c = new TCanvas("cdx", "Dx", 800, 600);
  hdata_dx_weight->SetStats(0);
  htred_dx_weight->SetStats(0);
  htred_dx->SetStats(0);

  // hdata_dx_weight->Draw();
  hdata_dx_weight->SetLineColor(kRed);
  hdata_dx_weight->GetYaxis()->SetRangeUser(0, hdata_dx_weight->GetMaximum() * 1.3);
  hdata_dx_weight->Draw("SAME");
  // htred_dx_weight->SetLineStyle(2);
  // htred_dx_weight->Draw("HIST SAME");
  htred_dx_weight->Draw("SAME");
  htred_dx_weight->SetLineColor(kBlue);
  htred_dx->SetLineStyle(2);
  htred_dx->Draw("HIST SAME");
  htred_dx->SetLineColor(kGreen+3);

  TLegend * leg = new TLegend(0.5, 0.7, 0.9, 0.9);
  leg->AddEntry(hdata_dx_weight.GetPtr(), hdata_dx_weight->GetTitle());
  leg->AddEntry(htred_dx_weight.GetPtr(), htred_dx_weight->GetTitle());
  leg->AddEntry(htred_dx.GetPtr(), htred_dx->GetTitle());
  leg->Draw();
  c->Print(::Form("comp_dx_weighted_totNmin%d_totNmax%d.png", totN_min, totN_max));
  delete c;
}


void weight_dx()
{
  weight_dx(1, 1);
  weight_dx(2, 2);
  weight_dx(3, 3);
  weight_dx(4, 4);
  weight_dx(1, 2);
  weight_dx(1, 3);
  weight_dx(1, 4);
  weight_dx(2, 3);
  weight_dx(2, 4);
  weight_dx(3, 4);
}
