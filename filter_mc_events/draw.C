#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TLegend.h>
#include <iostream>
#include <vector>
#include <string>

int draw() {
    std::vector<std::string> thres = {"5k", "8k", "10k", "12k"};
    std::vector<TH1F*> hists;

    TCanvas* c = new TCanvas("c", "Threshold Comparison", 800, 600);
    int nbins = 100;
    float qmin = 0;
    float qmax = 50;

    // Create histograms

    // Fill histograms
    for (size_t i = 0; i < thres.size(); ++i) {
        std::string filename = "nonoise_pid13_thres" + thres[i] + "_unipolar/many_muon_hits.root";
        TFile* file = TFile::Open(filename.c_str());
        if (!file || file->IsZombie()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            continue;
        }

        TTree* tree = (TTree*)file->Get("mu_ndlar/hits");
        if (!tree) {
            std::cerr << "Tree not found in file: " << filename << std::endl;
            file->Close();
            continue;
        }

        std::string hname = "hthres" + thres[i];
        std::string htitle = "Threshold " + thres[i];
        TH1F* h = new TH1F(hname.c_str(), htitle.c_str(), nbins, qmin, qmax);
        hists.push_back(h);

        std::string draw_cmd = "totQ >> hthres" + thres[i];
        std::string cut = "tindex == 0";

        std::cout << "Drawing: " << draw_cmd << " with cut: " << cut << std::endl;
        // int ien = tree->Draw("totQ", cut.c_str(), "goff");
        int ien = tree->Draw(draw_cmd.c_str(), cut.c_str(), "goff");
        std::cout << ien << std::endl;

        std::cout << tree->GetEntries() << std::endl;
        std::cout << "Histogram " << thres[i] << " entries: " << hists[i]->GetEntries() << std::endl;

        h->SetDirectory(0);

        file->Close();
    }

    // Plotting
    // c->SetLeftMargin(0.16);
    // c->SetRightMargin(0.04);
    // c->SetBottomMargin(0.16);
    hists[0]->SetTitle("Muon hits in TRED simulation");
    hists[0]->SetLineColor(kRed);
    hists[0]->SetStats(0);
    hists[0]->Draw();
    hists[0]->GetXaxis()->SetTitle("Total charge at trigger pixels [ke^{-}]");
    hists[0]->GetYaxis()->SetTitle("Pixel counts");
    hists[0]->GetXaxis()->CenterTitle();
    hists[0]->GetYaxis()->CenterTitle();

    int colors[] = {kBlue, kGreen + 2, kMagenta, kOrange + 7};
    for (size_t i = 1; i < hists.size(); ++i) {
        hists[i]->SetLineColor(colors[(i - 1) % 4]);
        hists[i]->Draw("SAME");
    }

    // Add legend
    TLegend* legend = new TLegend(0.6, 0.7, 0.88, 0.88);
    for (size_t i = 0; i < hists.size(); ++i) {
        legend->AddEntry(hists[i], ("Threshold " + thres[i]).c_str(), "l");
    }
    legend->Draw();

    c->Update();
    c->SaveAs("thres_comp_unipolar_pid13.png");

    return 0;
}

