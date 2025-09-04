#include "TTree.h"
#include "TH1D.h"


void draw_for_alex()
{
  // input files: track_data_0p1mm.root, track_data_1mm.root, track_data_normal.root
  // histogram: h("totQ", "totQ", 50, 0, 50);
  // tree: f->Get("selected_data/hits")
  // tree->Draw("totQ", "totQ < 50 && tindex == 0", "goff");
  TFile *f0 = TFile::Open("track_data_0p1mm.root");
  TFile *f1 = TFile::Open("track_data_1mm.root");
  TFile *f2 = TFile::Open("track_data_normal.root");
  TTree *t0 = (TTree*)f0->Get("selected_data/hits");
  TTree *t1 = (TTree*)f1->Get("selected_data/hits");
  TTree *t2 = (TTree*)f2->Get("selected_data/hits");
  TH1D *h0 = new TH1D("h0", "totQ for 0.1mm", 50, 0, 50);
  TH1D *h1 = new TH1D("h1", "totQ for 1mm", 50, 0, 50);
  TH1D *h2 = new TH1D("h2", "totQ for normal", 50, 0, 50);
  TCanvas *canvas = new TCanvas("canvas", "totQ Comparison", 800, 600);
  t0->Draw("totQ >> h0", "totQ < 50 && tindex == 0", "goff");
  t1->Draw("totQ >> h1", "totQ < 50 && tindex == 0", "goff");
  t2->Draw("totQ >> h2", "totQ < 50 && tindex == 0", "goff");
  h0->SetLineColor(kRed);
  h1->SetLineColor(kBlue);
  h2->SetLineColor(kGreen);
  h0->SetTitle("totQ for different track selections");
  h0->GetXaxis()->SetTitle("totQ per pixel (ke-)");
  h0->GetYaxis()->SetTitle("Counts");
  h0->Draw();
  h1->Draw("same");
  h2->Draw("same");
  TLegend *leg = new TLegend(0.7, 0.7, 0.9, 0.9);
  leg->AddEntry(h0, "0.1mm", "l");
  leg->AddEntry(h1, "1mm", "l");
  leg->AddEntry(h2, "normal", "l");
  leg->Draw();
  // gPad->SetLogy();
  gPad->Update();
  gStyle->SetOptStat(0);
  canvas->SaveAs("totQ_comparison.png");
  TFile *output = new TFile("totQ_comparison.root", "RECREATE");
  h0->Write();
  h1->Write();
  h2->Write();
  output->Close();
  f0->Close();
  f1->Close();
  f2->Close();
}
