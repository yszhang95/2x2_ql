#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TLegend.h"

double sum_total_length(std::string filename, std::string treename){
    auto df = ROOT::RDataFrame(treename, filename);
    return df.Sum("distance").GetValue();
}


void draw_totQ_mc(std::string fhits, std::string feffq, std::string label)
{
  TFile* f1 = TFile::Open(fhits.c_str());
  TFile* f2 = TFile::Open(feffq.c_str());
  if (f1->IsZombie() || f2->IsZombie()) {
    std::cerr << "Error opening file!" << std::endl;
    return;
  }
  auto t1 = f1->Get<TTree>("selected_data/hits");
  auto t2 = f2->Get<TTree>("selected_data/hits");
  auto h1 = new TH1F("h1","h1",50,0,50);
  auto h2 = new TH1F("h2","h2",50,0,50);
  t1->Draw("totQ>>h1","tindex==0");
  t2->Draw("totQ>>h2","tindex==0 & totQ>5");

  auto d1 = sum_total_length(fhits, "selected_data/distances");
  auto d2 = sum_total_length(feffq, "selected_data/distances");

  h1->Scale(1./d1);
  h2->Scale(1./d2);

  TCanvas* c1 = new TCanvas(("ctotQ"+label).c_str(), ("totQ"+label).c_str(), 800, 600);
  c1->SetRightMargin(0.05);
  c1->SetLeftMargin(0.15);
  c1->SetBottomMargin(0.15);
  h1->SetStats(0);
  h1->SetTitle("total Q per pixel;totQ;normalized counts per unit length");
  h1->GetYaxis()->SetRangeUser(0, 1.2* std::max(h1->GetMaximum(), h2->GetMaximum()));
  h1->Draw();
  h1->SetLineColor(kRed);
  h2->Draw("HIST SAME");
  h2->SetLineColor(kBlue);
  h2->SetLineStyle(2);
  TLegend * leg = new TLegend(0.7, 0.7, 0.95, 0.9);
  leg->AddEntry(h1, "Hits");
  leg->AddEntry(h2, "Ioni Q>5 (@anode)");
  leg->Draw();
  c1->Draw();
  c1->Print((label + "_comp_totQ_mc.png").c_str());
}

void draw_totQ_mc()
{
  draw_totQ_mc("merged_noshield_thres5k_hits.root", "merged_noshield_thres5k_effq.root", "noshield_thres5k");
  draw_totQ_mc("merged_shield_hits.root", "merged_shield_effq.root", "shield");
}
