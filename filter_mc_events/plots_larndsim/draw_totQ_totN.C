
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

double sum_total_length(std::string filename, std::string treename){
    auto df = ROOT::RDataFrame(treename, filename);
    return df.Sum("distance").GetValue();
}

void draw_dx(){
    auto hdx_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "dx", "", 100, -2,3);
    hdx_2x2->SetName("hdx_2x2");
    auto hdx_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "dx", "", 100,-2,3);
    hdx_larndsim->SetName("hdx_larndsim");
    TCanvas* c1 = new TCanvas("cdx", "dx", 800, 600);
    hdx_2x2->SetTitle(";dx;normalized counts");
    hdx_2x2->Scale(1./hdx_2x2->GetEntries());
    hdx_larndsim->Scale(1./hdx_larndsim->GetEntries());
    hdx_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hdx_2x2->GetMaximum(), hdx_larndsim->GetMaximum()));
    hdx_2x2->Draw();
    hdx_2x2->SetLineColor(kRed);
    hdx_larndsim->Draw("SAME");
    hdx_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hdx_2x2, "2x2");
    leg->AddEntry(hdx_larndsim, "larndsim MR6.4");
    leg->Draw();
    c1->Print("comp_dx.png");
}

void draw_totQ(const bool uselength){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0");
    htotQ_2x2->SetName("htotQ_2x2");
    auto htotQ_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totQ", "tindex == 0");
    htotQ_larndsim->SetName("htotQ_larndsim");
    TCanvas* c1 = new TCanvas("ctotQ", "totQ", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_larndsim->Scale(1./d_larndsim);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_larndsim->Scale(1./htotQ_larndsim->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_larndsim->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_larndsim->Draw("SAME");
    htotQ_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_larndsim, "larndsim MR6.4");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotQ_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (larndsim): %.2f * %.0fcm", htotQ_larndsim->Integral(), d_larndsim));
    }
    if (uselength) {
        c1->Print("comp_totQ_norm_by_l.png");
    } else {
        c1->Print("comp_totQ.png");
    }
}

void draw_totN(const bool uselength){

    auto htotN_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotN_2x2");
    auto htotN_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totN", "tindex == 0", 5, -0.5, 4.5);
    htotN_larndsim->SetName("htotN_larndsim");
    TCanvas* c2 = new TCanvas("ctotN", "totN", 800, 600);
    htotN_2x2->SetTitle("total N per pixel;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_larndsim->Scale(1./d_larndsim);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_larndsim->Scale(1./htotN_larndsim->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_larndsim->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_larndsim->Draw("SAME");
    htotN_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_larndsim, "larndsim 6.4");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (larndsim): %.2f * %.0fcm", htotN_larndsim->Integral(), d_larndsim));
    }
    if (uselength) {
        c2->Print("comp_totN_norm_by_l.png");
    } else {
        c2->Print("comp_totN.png");
    }
}

void draw_totN_totQ24(const bool uselength){

    auto htotN_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totN", "tindex == 0 && totQ>24", 5, -0.5, 4.5);
    htotN_2x2->SetName("htotNtotQ24_2x2");
    auto htotN_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totN", "tindex == 0 && totQ>24", 5, -0.5, 4.5);
    htotN_larndsim->SetName("htotNtotQ24_larndsim");
    TCanvas* c2 = new TCanvas("ctotN", "totN", 800, 600);
    htotN_2x2->SetTitle("total N per pixel,totQ>24;totN;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotN_2x2->Scale(1./d_2x2);
        htotN_larndsim->Scale(1./d_larndsim);
    } else {
        htotN_2x2->Scale(1./htotN_2x2->GetEntries());
        htotN_larndsim->Scale(1./htotN_larndsim->GetEntries());
    }
    htotN_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotN_2x2->GetMaximum(), htotN_larndsim->GetMaximum()));
    htotN_2x2->Draw();
    htotN_2x2->SetLineColor(kRed);
    htotN_larndsim->Draw("SAME");
    htotN_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotN_2x2, "2x2");
    leg->AddEntry(htotN_larndsim, "larndsim MR6.4");
    leg->Draw();
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.5, 0.55, ::Form("Integral (2x2): %.2f * %.0fcm", htotN_2x2->Integral(), d_2x2));
        tex->DrawLatexNDC(0.5, 0.5, ::Form("Integral (larndsim): %.2f * %.0fcm", htotN_larndsim->Integral(), d_larndsim));
    }
    if (uselength) {
        c2->Print("comp_totNtotQ24_norm_by_l.png");
    } else {
        c2->Print("comp_totNtotQ24.png");
    }
}

void draw_totQ_totN1(const bool uselength){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0 && totN == 1");
    htotQ_2x2->SetName("htotQ_2x2_totN1");
    auto htotQ_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totQ", "tindex == 0 && totN==1");
    htotQ_larndsim->SetName("htotQ_larndsim_totN1");
    TCanvas* c1 = new TCanvas("ctotQtotN1", "totQtotN1", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==1;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_larndsim->Scale(1./d_larndsim);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_larndsim->Scale(1./htotQ_larndsim->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_larndsim->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_larndsim->Draw("SAME");
    htotQ_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_larndsim, "larndsim MR6.4");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (larndsim): %.2f", htotQ_larndsim->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN1_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN1.png");
    }
}

void draw_totQ_totN2(const bool uselength){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0 && totN == 2");
    htotQ_2x2->SetName("htotQ_2x2_totN2");
    auto htotQ_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totQ", "tindex == 0 && totN==2");
    htotQ_larndsim->SetName("htotQ_larndsim_totN2");
    TCanvas* c1 = new TCanvas("ctotQtotN2", "totQtotN2", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==2;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_larndsim->Scale(1./d_larndsim);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_larndsim->Scale(1./htotQ_larndsim->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_larndsim->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_larndsim->Draw("SAME");
    htotQ_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_larndsim, "larndsim MR 6.4");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (larndsim): %.2f", htotQ_larndsim->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN2_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN2.png");
    }
}

void draw_totQ_totN3(const bool uselength){

    auto htotQ_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "totQ", "tindex == 0 && totN == 2");
    htotQ_2x2->SetName("htotQ_2x2_totN3");
    auto htotQ_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "totQ", "tindex == 0 && totN==2");
    htotQ_larndsim->SetName("htotQ_larndsim_totN3");
    TCanvas* c1 = new TCanvas("ctotQtotN3", "totQtotN3", 800, 600);
    htotQ_2x2->SetTitle("total Q per pixel, totN==3;totQ;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        htotQ_2x2->Scale(1./d_2x2);
        htotQ_larndsim->Scale(1./d_larndsim);
    } else {
        htotQ_2x2->Scale(1./htotQ_2x2->GetEntries());
        htotQ_larndsim->Scale(1./htotQ_larndsim->GetEntries());
    }
    htotQ_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(htotQ_2x2->GetMaximum(), htotQ_larndsim->GetMaximum()));
    htotQ_2x2->Draw();
    htotQ_2x2->SetLineColor(kRed);
    htotQ_larndsim->Draw("SAME");
    htotQ_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(htotQ_2x2, "2x2");
    leg->AddEntry(htotQ_larndsim, "larndsim MR6.4");
    leg->Draw();
    if (uselength) {
        TLatex *tex = new TLatex();
        tex->SetTextFont(42);
        tex->DrawLatexNDC(0.2, 0.75, ::Form("Integral (2x2): %.2f", htotQ_2x2->Integral()));
        tex->DrawLatexNDC(0.2, 0.68, ::Form("Integral (larndsim): %.2f", htotQ_larndsim->Integral()));
    }
    if (uselength) {
        c1->Print("comp_totQ_totN3_norm_by_l.png");
    } else {
        c1->Print("comp_totQ_totN3.png");
    }
}

void draw_thres(const bool uselength){

    auto hthres_2x2 = draw_from_tree("../track_data.root", "selected_data/hits", "thres", "tindex == 0");
    hthres_2x2->SetName("hthres_2x2");
    auto hthres_larndsim = draw_from_tree("../track_larndsim_mr6p4.root", "selected_data/hits", "thres", "tindex == 0");
    hthres_larndsim->SetName("hthres_larndsim");
    TCanvas* c1 = new TCanvas("cthres", "thres", 800, 600);
    hthres_2x2->SetTitle("frequency of thresholds at each triggered channel;thres;normalized counts");
    if (uselength) {
        auto d_2x2 = sum_total_length("../track_data.root", "selected_data/distances");
        auto d_larndsim = sum_total_length("../track_larndsim_mr6p4.root", "selected_data/distances");
        hthres_2x2->Scale(1./d_2x2);
        hthres_larndsim->Scale(1./d_larndsim);
    } else {
        hthres_2x2->Scale(1./hthres_2x2->GetEntries());
        hthres_larndsim->Scale(1./hthres_larndsim->GetEntries());
    }
    hthres_2x2->GetYaxis()->SetRangeUser(0, 1.2* std::max(hthres_2x2->GetMaximum(), hthres_larndsim->GetMaximum()));
    hthres_2x2->Draw();
    hthres_2x2->SetLineColor(kRed);
    hthres_larndsim->Draw("SAME");
    hthres_larndsim->SetLineColor(kBlue);
    TLegend * leg = new TLegend(0.5, 0.7, 0.8, 0.85);
    leg->AddEntry(hthres_2x2, "2x2");
    leg->AddEntry(hthres_larndsim, "larndsim MR6.4");
    leg->Draw();
    if (uselength) {
        c1->Print("comp_thres_norm_by_l.png");
    } else {
        c1->Print("comp_thres.png");
    }
}

void draw_totQ_totN(bool uselength=false){
    gStyle->SetOptStat(0);
    draw_totQ(uselength);
    draw_totN(uselength);
    draw_totN_totQ24(uselength);
    draw_totQ_totN1(uselength);
    draw_totQ_totN2(uselength);
    draw_totQ_totN3(uselength);
    draw_thres(uselength);
    draw_dx();
}
