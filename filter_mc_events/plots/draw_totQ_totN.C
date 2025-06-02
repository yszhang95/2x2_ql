
TH1F* draw_from_tree(std::string filename, std::string treename, std::string var, std::string sel, int n=45, float nmin=0, float nmax=45) {
    // Load the ROOT file
    TFile *file = TFile::Open(filename.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return nullptr;
    }

    // Get the tree from the file
    TTree *tree = (TTree*)file->Get(treename.c_str());
    if (!tree) {
        std::cerr << "Error: Tree not found!" << std::endl;
        return nullptr;
    }

    // Draw the 2D histogram
    TH1F *h1d = new TH1F("h1d", "", n, nmin, nmax);
    tree->Draw(::Form("%s>>h1d", var.c_str()), sel.c_str());
    h1d->SetDirectory(0);
    file->Close();
    return h1d;
}

void draw_dx(){
    auto hdx_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "dx", "", 100, -2,3);
    hdx_2x2->SetName("hdx_2x2");
    auto hdx_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "dx", "", 100,-2,3);
    hdx_tred->SetName("hdx_tred");
    TCanvas* c1 = new TCanvas("cdx", "dx", 800, 600);
    hdx_2x2->SetTitle(";dx;normalized counts");
    hdx_2x2->Scale(1./hdx_2x2->GetEntries());
    hdx_tred->Scale(1./hdx_tred->GetEntries());
    hdx_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hdx_2x2->GetMaximum(), hdx_tred->GetMaximum()));
    hdx_2x2->Draw();
    hdx_2x2->SetLineColor(kRed);
    hdx_tred->Draw("SAME");
    hdx_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hdx_2x2, "2x2");
    leg->AddEntry(hdx_tred, "tred");
    leg->Draw();
    c1->Print("comp_dx.png");
}

void draw_totQ(){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0");
    htotQ_2x2->SetName("htotQ_2x2");
    auto htotQ_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "totQ", "tindex == 0");
    htotQ_tred->SetName("htotQ_tred");
    TCanvas* c1 = new TCanvas("ctotQ", "totQ", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel;totQ;normalized counts");
    htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
    htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    c1->Print("comp_totQ.png");
}

void draw_totN(){

    auto htotN_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotN_2x2");
    auto htotN_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_tred->SetName("htotN_tred");
    TCanvas* c2 = new TCanvas("ctotN", "totN", 800, 600);
    htotN_2x2->SetTitle("total Q per pixel;totN;normalized counts");
    htotN_2x2->Scale(1./htotN_2x2->GetEntries());
    htotN_tred->Scale(1./htotN_tred->GetEntries());
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_tred->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_tred->Draw("SAME");
    htotN_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_tred, "tred");
    leg->Draw();
    c2->Print("comp_totN.png");
}

void draw_totQ_totN1(){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0 && totN == 1");
    htotQ_2x2->SetName("htotQ_2x2_totN1");
    auto htotQ_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "totQ", "tindex == 0 && totN==1");
    htotQ_tred->SetName("htotQ_tred_totN1");
    TCanvas* c1 = new TCanvas("ctotQtotN1", "totQtotN1", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==1;totQ;normalized counts");
    htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
    htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    c1->Print("comp_totQ_totN1.png");
}

void draw_totQ_totN2(){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0 && totN == 2");
    htotQ_2x2->SetName("htotQ_2x2_totN2");
    auto htotQ_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "totQ", "tindex == 0 && totN==2");
    htotQ_tred->SetName("htotQ_tred_totN2");
    TCanvas* c1 = new TCanvas("ctotQtotN2", "totQtotN2", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==2;totQ;normalized counts");
    htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
    htotQ_tred->Scale(1./htotQ_tred->GetEntries());
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_tred->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_tred->Draw("SAME");
    htotQ_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_tred, "tred");
    leg->Draw();
    c1->Print("comp_totQ_totN2.png");
}

void draw_thres(){

    auto hthres_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "thres", "tindex == 0");
    hthres_2x2->SetName("hthres_2x2");
    auto hthres_tred = draw_from_tree("../lifetime1ms/many_muon_hits.root", "mu_ndlar/hits", "thres", "tindex == 0");
    hthres_tred->SetName("hthres_tred");
    TCanvas* c1 = new TCanvas("cthres", "thres", 800, 600);
    hthres_2x2->SetTitle("frequency of thresholds at each triggered channel;thres;normalized counts");
    hthres_2x2->Scale(1./hthres_2x2->GetEntries());
    hthres_tred->Scale(1./hthres_tred->GetEntries());
    hthres_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hthres_2x2->GetMaximum(), hthres_tred->GetMaximum()));
    hthres_2x2->Draw();
    hthres_2x2->SetLineColor(kRed);
    hthres_tred->Draw("SAME");
    hthres_tred->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hthres_2x2, "2x2");
    leg->AddEntry(hthres_tred, "tred");
    leg->Draw();
    c1->Print("comp_thres.png");
}

void draw_totQ_totN(){
    gStyle->SetOptStat(0);
    draw_totQ();
    draw_totN();
    draw_totQ_totN1();
    draw_totQ_totN2();
    draw_thres();
    draw_dx();
}
