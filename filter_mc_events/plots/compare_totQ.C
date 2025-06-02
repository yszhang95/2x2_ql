#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>

void compare_totQ() {
    // Open files
    TFile* file1 = TFile::Open("track_data.root");
    TFile* file2 = TFile::Open("track_tred.root");

    if (!file1 || file1->IsZombie() || !file2 || file2->IsZombie()) {
        std::cerr << "Failed to open one or both files!" << std::endl;
        return;
    }

    // Get trees
    TTree* tree1 = (TTree*)file1->Get("selected_data/hits");  // Replace "tree_path"
    TTree* tree2 = (TTree*)file2->Get("selected_data/hits");

    if (!tree1 || !tree2) {
        std::cerr << "Tree not found in one of the files." << std::endl;
        return;
    }

    // Create histograms
    TH1F* h1 = new TH1F("h1", "total Q per pixel from MIP tracks;totoal charge per pixel [ke-];Normalized Counts", 50, 0, 50);
    TH1F* h2 = new TH1F("h2", "total Q per pixel from MIP tracks;totoal charge per pixel [ke-];Normalized Counts", 50, 0, 50);

    // Draw with selection
    tree1->Draw("totQ >> h1", "tindex == 0", "goff");
    tree2->Draw("totQ >> h2", "tindex == 0", "goff");

    // Normalize
    // if (h1->Integral() > 0) h1->Scale(1.0 / h1->Integral(0, 100000));
    // if (h2->Integral() > 0) h2->Scale(1.0 / h2->Integral(0, 100000));
    h1->Scale(1./h1->GetEntries());
    h2->Scale(1./h2->GetEntries());

    float max1 = h1->GetMaximum();
    float max2 = h2->GetMaximum();

    float max_total = std::max(max1, max2);
    std::cout << "Maximum bin height: " << max_total << std::endl;

    // Style
    h1->SetLineColor(kRed);
    h2->SetLineColor(kBlue);

    // Canvas and plot
    TCanvas* c = new TCanvas("c", "totQ Comparison", 800, 600);
    // h1->Draw("HIST");
    // h2->Draw("HIST SAME");
    h1->Draw("E");
    h2->Draw("E SAME");
    h1->GetYaxis()->SetRangeUser(0, max_total*1.2);
    h1->SetStats(0);

    h1->GetXaxis()->SetTitleSize(0.05);
    h1->GetYaxis()->SetTitleSize(0.05);
    c->SetLeftMargin(0.14);
    c->SetBottomMargin(0.14);
    c->SetRightMargin(0.06);

    // Legend
    TLegend* legend = new TLegend(0.6, 0.7, 0.88, 0.88);
    legend->AddEntry(h1, "2x2 data", "l");
    legend->AddEntry(h2, "TRED", "l");
    legend->Draw();

    c->Print("compare_totQ.png");
    c->Print("compare_totQ.pdf");
}

