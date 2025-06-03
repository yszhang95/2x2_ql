std::pair<TH2F*, TH1F*> get_hists(std::string fname, std::string treename, std::string label){
    // Load the ROOT file
    TFile *file = TFile::Open(fname.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return {nullptr, nullptr};
    }

    // Get the tree from the file
    TTree *tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
        std::cerr << "Error: Tree not found!" << std::endl;
        return {nullptr, nullptr};
    }

    // Create a canvas
    TCanvas *c1 = new TCanvas(::Form("c1_%s", label.c_str()), "AvgI vs Dx", 800, 600);
    c1->SetRightMargin(0.15); // Leave room for color palette

    // Draw the 2D histogram
    TH2F *h2d = new TH2F("h2d", "AvgI vs Dx", 50, -2, 3, 50, -0.5, 2.);
    tree->Draw("avg_i:dx>>h2d", "", "COLZ");

    // Customize histogram
    h2d->SetTitle(::Form("Hits from MIP tracks, %s", label.c_str()));
    h2d->GetXaxis()->SetTitle("dx [cm]");
    h2d->GetYaxis()->SetTitle("Average Current [ke^{-}/0.1#mus]");
    h2d->GetXaxis()->CenterTitle();
    h2d->GetYaxis()->CenterTitle();
    h2d->SetStats(0);
    h2d->SetDirectory(0);
    h2d->SetName(::Form("h2d_%s", label.c_str()));

    // Update and save
    c1->SetLogz();
    c1->Update();
    c1->Print(::Form("AvgCurrVsDx_%s.png", label.c_str()));  // Save as image

    h2d->SetDirectory(0);
    file->Close();

    TCanvas* c2 = new TCanvas(::Form("c2_%s", label.c_str()), "AvgI profile", 600, 400);
    TH1F* h1d = new TH1F(::Form("h1d_%s", label.c_str()), "", h2d->GetXaxis()->GetNbins(), h2d->GetXaxis()->GetBinLowEdge(0), h2d->GetXaxis()->GetBinLowEdge(h2d->GetXaxis()->GetNbins()+1));
    for (int i=1; i<h2d->GetNbinsX()+1; ++i) {
      auto h = h2d->ProjectionY(::Form("hbin%d", i), i, i);
      auto mean = h->GetMean();
      auto rms = h->GetRMS();
      auto stddev = h->GetStdDev();
      if (h->GetEntries()>10) {
        h1d->SetBinContent(i, mean);
        h1d->SetBinError(i, rms/std::sqrt(h->GetEntries()));
      }
    }
    c2->Draw();
    h1d->SetTitle(::Form("Profile along dx. Entries < 10 filtered, %s", label.c_str()));
    h1d->GetXaxis()->SetTitle("dx [cm]");
    h1d->GetYaxis()->SetTitle("Average current [ke^{-1}/0.1#mus]");
    h1d->SetStats(0);
    h1d->GetXaxis()->CenterTitle();
    h1d->GetYaxis()->CenterTitle();
    h1d->Draw("E");
    c2->Print(::Form("AvgCurrVsDxProfile_%s.png", label.c_str()));
    return {h2d, h1d};
}

void draw_AvgCurrVsDx(){
                                   //
    auto hists_2x2 = get_hists("../track_data.root", "selected_data/hits", "2x2");
    auto hists_tred = get_hists("../lifetime1ms/many_muon_hits_selected.root", "mu_ndlar/hits", "tred");
    
}

