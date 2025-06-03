void draw_QvsThres(std::string filename, std::string treename, std::string label, int totN=1) {
    // Load the ROOT file
    TFile *file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Get the tree from the file
    TTree *tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
        std::cerr << "Error: Tree not found!" << std::endl;
        return;
    }

    // Create a canvas
    TCanvas *c1 = new TCanvas("c1", "Q vs Thres", 800, 600);
    c1->SetRightMargin(0.15); // Leave room for color palette

    // Draw the 2D histogram
    TH2F *h2d = new TH2F("h2d", "Q vs Thres", 30, 0, 30, 45, 0, 45);
    tree->Draw("Q:thres>>h2d", ::Form("totN==%d", totN), "COLZ");
    h2d->SetDirectory(0);
    file->Close();

    TF1* f = new TF1("f", "x", 0, 50);
    f->SetLineStyle(2);
    f->Draw("SAME");

    // Customize histogram
    h2d->SetTitle(::Form("Q vs Thres. for number of hits per pixel = %d, %s", totN, label.c_str()));
    h2d->GetXaxis()->SetTitle("Threshold [ke]");
    h2d->GetYaxis()->SetTitle("Total Charge per pixel (ke)");
    h2d->GetXaxis()->CenterTitle();
    h2d->GetYaxis()->CenterTitle();
    h2d->SetStats(0);

    // Update and save
    // c1->SetLogz();
    c1->Update();
    c1->Print(::Form("QvsThres_totN%d_%s.png", totN, label.c_str()));  // Save as image
    // c1->Print(::Form("QvsThres_totN%d_%s.pdf", totN, label.c_str()));  // Save as image
}

void draw_QvsThres_totN()
{

    draw_QvsThres("../track_data.root", "selected_data/hits", "2x2", 1);
    draw_QvsThres("../track_data.root", "selected_data/hits", "2x2", 2);

    draw_QvsThres("../lifetime1ms/many_muon_hits_selected.root", "mu_ndlar/hits", "tred", 1);
    draw_QvsThres("../lifetime1ms/many_muon_hits_selected.root", "mu_ndlar/hits", "tred", 2);
}
